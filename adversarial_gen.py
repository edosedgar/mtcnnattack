from __future__ import division, print_function, absolute_import
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import imageio
import cv2

import mtcnn.mtcnn as mtcnn
import utils.inter_area as inter_area
import utils.patch_mng as patch_mng

import os
import shutil
import time

# ===================================================
# Define class for training procedure
# ===================================================

class TrainMask:
    def __init__(self, gpu_id=4):
        self.pm = patch_mng.PatchManager()
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
        self.compiled = 0
        self.masks_tf = []
        self.sizes = []
        self.eps = tf.Variable(initial_value=0, dtype=tf.float32)
        self.mu = tf.Variable(initial_value=0, dtype=tf.float32)
        self.accumulators = []

    # ===================================================
    # All masks should be within 0..255 range otherwise will be clipped
    # Color: RGB
    # NOTE: the mask itself can be either b/w or color (HxWx1 or HxWx3)
    # ===================================================
    def add_masks(self, masks):
        for key in masks.keys():
            data = masks[key]
            mask = self.pm.add_patch(data[0].clip(0, 255),
                                     key, data[1][::-1], data[2][::-1])
            self.masks_tf.append(mask)

    # ===================================================
    # All images should be located in 'input_img' directory
    # ===================================================
    def add_images(self, images):
        for filename in images:
            img = cv2.imread("input_img/" + filename, cv2.IMREAD_COLOR)
            self.pm.add_image(img)

    # ===================================================
    # Here all TF variables will be prepared
    # The method could be re-run to restore initial values
    # ===================================================
    def build(self, sess):
        if (self.compiled == 0):
            self.pm.compile()
            self.init = self.pm.prepare_imgs()
            self.init_vars = self.pm.init_vars()
            for i, key in enumerate(self.pm.patches.keys()):
                mask_tf = self.pm.patches[key].mask_tf
                accumulator = tf.Variable(tf.zeros_like(mask_tf))
                self.accumulators.append(accumulator)
            self.init_accumulators = tf.initializers.variables(self.accumulators)
            self.compiled = 1

        sess.run(self.init_vars)
        sess.run(self.init)
        sess.run(self.init_accumulators)

    # ===================================================
    # Set the sizes pictures will be scaled
    # ===================================================
    def set_input_sizes(self, sizes):
        self.sizes = sizes

    # ===================================================
    # Here the batch of images will be resized, transposed and normalized
    # ===================================================
    def scale(self, imgs, h, w):
        scaled = inter_area.resize_area_batch(tf.cast(imgs, tf.float64), h, w)
        transposed = tf.transpose(tf.cast(scaled, tf.float32), (0, 2, 1, 3))
        normalized = ((transposed * 255) - 127.5) * 0.0078125
        return normalized

    # ===================================================
    # Build up training function to be used for attacking
    # ===================================================
    def build_train(self, sess, config):
        size2str = (lambda size: str(size[0]) + "x" + str(size[1]))
        pnet_loss = []
        patch_loss = []
        eps = self.eps
        mu = self.mu
        mask_assign_op = []
        moment_assign_op = []

        # Apply all patches and augment
        img_w_mask = self.pm.apply_patches(config.colorizer_wb2rgb)
        self.img_hat = img_w_mask
        noise = tf.random_normal(shape=tf.shape(img_w_mask), mean=0.0, stddev=0.02, dtype=tf.float32)
        img_w_mask = tf.clip_by_value(img_w_mask + noise, 0.0, 1.0)

        # Create PNet for each size and calc PNet probability map loss
        for size in self.sizes:
            img_scaled = self.scale(img_w_mask, size[0], size[1])
            with tf.variable_scope('pnet_' + size2str(size), reuse=tf.AUTO_REUSE):
                pnet = mtcnn.PNet({'data': img_scaled}, trainable=False)
                pnet.load(os.path.join("./weights", 'det1.npy'), sess)
                clf = sess.graph.get_tensor_by_name("pnet_" + size2str(size) + "/prob1:0")
                bb = sess.graph.get_tensor_by_name("pnet_" + size2str(size) + "/conv4-2/BiasAdd:0")
                pnet_loss.append(config.apply_pnet_loss(clf, bb))
        pnet_loss_total = tf.add_n(pnet_loss)

        # Calculate loss for each patch and do FGSM
        for i, key in enumerate(self.pm.patches.keys()):
            mask_tf = self.pm.patches[key].mask_tf

            multiplier = tf.cast((eps <= 55/255.0), tf.float32)
            patch_loss_total = multiplier * config.apply_patch_loss(mask_tf, i, key)
            total_loss = tf.identity(pnet_loss_total + patch_loss_total, name="total_loss")

            grad_raw = tf.gradients(total_loss, mask_tf)[0]
            new_moment = mu * self.accumulators[i] + grad_raw / tf.norm(grad_raw, ord=1)
            assign_op1 = tf.assign(self.accumulators[i], new_moment) 
            moment_assign_op.append(assign_op1)
            new_mask = tf.clip_by_value(mask_tf - eps * tf.sign(self.accumulators[i]), 0.0, 1.0)
            assign_op2 = tf.assign(self.pm.patches[key].mask_tf, new_mask)
            mask_assign_op.append(assign_op2)

        # Return assign operation for each patch
        self.mask_assign_op = tuple(mask_assign_op)
        self.moment_assign_op = tuple(moment_assign_op)

    # ===================================================
    # Schedule *learning rate* so that opt process gets better
    # ===================================================
    def lr_schedule(self, i):
        if (i < 100):
            feed_dict = {self.eps: 60/255.0, self.mu: 0.9}
        if (i >= 100 and i < 300):
            feed_dict = {self.eps: 30/255.0, self.mu: 0.9}
        if (i >= 300 and i < 1000):
            feed_dict = {self.eps: 15/255.0, self.mu: 0.95}
        if (i >= 1000):
            feed_dict = {self.eps: 1/255.0,  self.mu: 0.99}
        return feed_dict

    def train(self, sess, i):
        feed_dict = self.lr_schedule(i)
        sess.run(self.moment_assign_op, feed_dict=feed_dict)
        sess.run(self.mask_assign_op, feed_dict=feed_dict)

    # ===================================================
    # Set of aux functions to be used for evaluating and init
    # ===================================================
    def eval(self, sess, dir):
        path_info = "output_img/" + dir + "/"
        shutil.rmtree(path_info, ignore_errors=True)
        os.makedirs(path_info)
        self.eval_masks(sess, path_info)
        self.eval_img(sess, path_info)

    def eval_masks(self, sess, dir):
        for key in self.pm.patches.keys():
            mask_tf = self.pm.patches[key].mask_tf
            mask = (mask_tf.eval(session=sess) * 255).astype(np.uint8)
            imageio.imsave(dir + key + ".png", mask)

    def eval_img(self, sess, dir):
        width = int(self.pm.imgs_tf.shape[2])
        bs = int(self.pm.imgs_tf.shape[0])
        imgs = (self.img_hat.eval(session=sess) * 255).astype(np.uint8)
        for i in range(bs):
            img = imgs[i]
            imageio.imsave(dir + "attacked" + str(i + 1) + ".png", img)

# ===================================================
# $$$$$ Define class for loss manipulation $$$$$$$$$$
# ===================================================
class LossManager:
    def __init__(self):
        self.patch_loss = {}
        self.pnet_loss = {}

    # ===================================================
    # Loss function for classification layer output
    # ===================================================

    # (minimize the max value of output prob map)
    def clf_loss_max(self, clf, bb):
        out = tf.reduce_max(tf.math.maximum(clf[...,1] - 0.5, 0.0), axis=(1, 2))
        return tf.reduce_mean(out)

    # (minimize the sum of squares from output prob map)
    def clf_loss_l2(self, clf, bb):
        out = tf.reduce_sum(tf.math.maximum(clf[...,1] - 0.5, 0.0) ** 2, axis=(1, 2))
        return tf.reduce_mean(out)

    # (minimize the sum of the absolute differences for neighboring pixel-values)
    def tv_loss(self, patch):
        loss = tf.image.total_variation(patch)
        return loss

    # (minimize the area with black color)
    def white_loss(self, patch):
        loss = tf.reduce_sum((1 - patch) ** 2)
        return loss

    # ===================================================
    # Input HxWxC
    # ===================================================
    def reg_patch_loss(self, func, name, coefs):
        self.patch_loss[name] = { 'func' : func, 'coef': coefs }

    # ===================================================
    # Input BSxPHxPW
    # ===================================================
    def reg_pnet_loss(self, func, name, coef):
        self.pnet_loss[name] = { 'func' : func, 'coef': coef }

    # ===================================================
    # Apply losses
    # ===================================================
    def apply_patch_loss(self, patch, patch_i, key):
        patch_loss = []
        for loss in self.patch_loss.keys():
            with tf.variable_scope(loss):
                c = self.patch_loss[loss]['coef'][patch_i]
                patch_loss.append(c * self.patch_loss[loss]['func'](patch))
            tf.summary.scalar(loss + "/" + key, c * patch_loss[-1])
        return tf.add_n(patch_loss)
    
    def apply_pnet_loss(self, clf, bb):
        pnet_loss = []
        for loss in self.pnet_loss.keys():
            with tf.variable_scope(loss):
                c = self.pnet_loss[loss]['coef']
                pnet_loss.append(c * self.pnet_loss[loss]['func'](clf, bb))
            tf.summary.scalar(loss, c * pnet_loss[-1])
        return tf.add_n(pnet_loss)

    def colorizer_wb2rgb(self, patch):
        return tf.image.grayscale_to_rgb(patch)

masks = {
    'left_cheek': [np.zeros((150, 180, 1)), (0, 255, 0), (0, 1, 0)],
    'right_cheek': [np.zeros((150, 180, 1)), (255, 0, 0), (1, 0, 0)],
}
images = ['1.png', '2.png']

config = LossManager()
tf.reset_default_graph()
adv_mask = TrainMask(gpu_id=2)
adv_mask.add_masks(masks)
adv_mask.add_images(images)
sess = tf.Session()

epochs = 2000
config.reg_pnet_loss(config.clf_loss_l2, 'clf_max', 1)
config.reg_patch_loss(config.tv_loss, 'tv_loss', [1e-5, 1e-5])

# Do not forget to analyze the sizes that are suitable for
# your resolution
adv_mask.set_input_sizes([(73, 129), (103, 182), (52, 92)])
adv_mask.build(sess)
adv_mask.build_train(sess, config)

for i in range(epochs):
    print(str(i + 1) + "/" + str(epochs), end='\r')
    adv_mask.train(sess, i)

adv_mask.eval(sess, "")
sess.close()