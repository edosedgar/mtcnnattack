from __future__ import division, print_function, absolute_import
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
import imageio
import copy
import os
import time

class PatchPartTF:
    def __init__(self):
        self.transforms = []

    def add_transform(self, matrix):
        self.transforms.append(matrix)

    def get_transform(self, part_num):
        return self.transforms[part_num]

class ImageTF:
    def __init__(self, img):
        self.image = copy.deepcopy(img.copy())

    def get_image(self):
        return self.image

    def get_grid_size(self, init_clr, grad_clr):
        clr = np.array(init_clr)
        dot_map = cv2.inRange(self.image, clr, clr)
        points, _ = np.where(dot_map > 0)
        width = len(points)
        if (width == 0):
            return
        height = 1
        while True:
            clr = clr - np.array(grad_clr)
            dot_map = cv2.inRange(self.image, clr, clr)
            points, _ = np.where(dot_map > 0)
            if (width == len(points)):
                height = height + 1
            else:
                break
        return (height, width)

    def add_new_line(self, xx, yy, dots_mask, i):
        # Find coordinates for the line
        y, x = np.where(dots_mask > 0)
        points = np.array(list(zip(y, x)))
        points = np.array(sorted(points, key=lambda row: row[1]))
        xx[i,:], yy[i,:] = points[:,1], points[:,0]
        return xx, yy

    def get_grid(self, init_clr, grad_clr, grid_size):
        # Create array with coordinates for destination mask
        clr = init_clr
        x, y = np.arange(grid_size[1]), np.arange(grid_size[0])
        xx_dst, yy_dst = np.meshgrid(x, y)
        for i in range(grid_size[0]):
            dots_mask = cv2.inRange(self.image, clr, clr)
            xx_dst, yy_dst = self.add_new_line(xx_dst, yy_dst, dots_mask, i)
            clr = clr - np.array(grad_clr)
        return xx_dst, yy_dst

class PatchTF:
    def __init__(self, mask, key, init_clr, grad_clr):
        self.mask_np = copy.deepcopy(mask.copy())
        self.mask_tf = tf.Variable(initial_value=mask, name=key, dtype=tf.float32)
        self.init_clr = init_clr
        self.grad_clr = grad_clr
        self.grid_size = 0
        self.parts = {}
        self.src_masks_np = []
        self.src_masks_tf = []
        self.key = key
        self.batch_transform = []

    def set_grid(self, image):
        self.grid_size = image.get_grid_size(self.init_clr, self.grad_clr)

    def add_part_transform(self, quad1, quad2, img_num):
        if img_num not in self.parts.keys():
            patch_part = PatchPartTF()
        else:
            patch_part = self.parts[img_num]
        M = cv2.getPerspectiveTransform(np.float32(quad2), np.float32(quad1))
        patch_part.add_transform(M)
        self.parts[img_num] = patch_part

    def pack_transforms(self):
        final_tr = []
        for i, part in enumerate(self.src_masks_np):
            part_batch_tr = np.ones((len(self.parts), 8))
            for img_tr in self.parts:
                M = self.parts[img_tr].get_transform(i)
                M = np.delete(np.append(M, []), -1)
                part_batch_tr[img_tr] = M
            final_tr.append(part_batch_tr)
        self.batch_transform = final_tr

    def get_quad(self, yy, xx, v, h):
        # Get 4 points of quadrangle
        return [[xx[v, h], yy[v, h]], [xx[v, h + 1], yy[v, h + 1]],
                [xx[v + 1, h], yy[v + 1, h]], [xx[v + 1, h + 1], yy[v + 1, h + 1]]]

    def pad_mask_np(self, img, mask, mask_pos):
        mask_padded = np.pad(mask, [[mask_pos[0][1], img.shape[0] - mask_pos[1][1]],\
                                    [mask_pos[0][0], img.shape[1] - mask_pos[1][0]],
                                    [0, 0]], mode="constant", constant_values=0)
        return mask_padded

    def overlap_quad(self, pts, v, h, dsize, typ):
        # Prolong the width and height to avoid gaps between subpatches
        if (v == (dsize[0] - 2) and typ != 0):
            pts[2][1] = pts[2][1] + 1
            pts[3][1] = pts[3][1] + 1
        if (h == (dsize[1] - 2) and typ != 0):
            pts[1][0] = pts[1][0] + 1
            pts[3][0] = pts[3][0] + 1
        return pts

    def init_src_parts(self):
        # Create array with coordinates for source mask
        x = np.linspace(0, self.mask_np.shape[1], self.grid_size[1]).astype(np.int32)
        y = np.linspace(0, self.mask_np.shape[0], self.grid_size[0]).astype(np.int32)
        xx_src, yy_src = np.meshgrid(x, y)
        part_padded = np.ones_like(self.mask_np)
        for v in range(self.grid_size[0] - 1):
            for h in range(self.grid_size[1] - 1):
                # Get coordinates of source patch and destination
                pts1 = self.get_quad(yy_src, xx_src, v, h)
                # Leave only the part to be transformed
                part = np.ones((pts1[3][1] - pts1[0][1], pts1[3][0] - pts1[0][0], self.mask_np.shape[2]))
                part_pad = self.pad_mask_np(part_padded, part, [pts1[0], pts1[3]])
                mask_part_tf = tf.Variable(initial_value=part_pad, name=self.key + "_src_mask",
                                           dtype=tf.float32)
                self.src_masks_np.append(copy.deepcopy(part_pad))
                self.src_masks_tf.append(mask_part_tf)

    def init_transforms(self, images):
        # Create array with coordinates for source mask
        x = np.linspace(0, self.mask_np.shape[1], self.grid_size[1]).astype(np.int32)
        y = np.linspace(0, self.mask_np.shape[0], self.grid_size[0]).astype(np.int32)
        xx_src, yy_src = np.meshgrid(x, y)
        for i, image in enumerate(images):
            xx_dst, yy_dst = image.get_grid(self.init_clr, self.grad_clr, self.grid_size)
            for v in range(self.grid_size[0] - 1):
                for h in range(self.grid_size[1] - 1):
                    # Get coordinates of source patch and destination
                    pts1 = self.get_quad(yy_src, xx_src, v, h)
                    pts2 = self.get_quad(yy_dst, xx_dst, v, h)
                    # Slightly overlap coordinates
                    pts2 = self.overlap_quad(pts2, v, h, self.grid_size, 1)
                    self.add_part_transform(pts1, pts2, i)

class PatchManager:
    def __init__(self):
        self.patches = {}
        self.images = []
        self.imgs_tf = None
        self.stencil = None

    def pad_img_tf(self, img, mask):
        paddings = tf.constant([[0, int(img.shape[0] - mask.shape[0])],
                                [0, int(img.shape[1] - mask.shape[1])],
                                [0, 0]])
        mask_padded = tf.pad(mask, paddings, mode="CONSTANT", constant_values=0.0)
        return mask_padded

    def batchify(self, elem):
        return tf.stack([elem for i in range(len(self.images))], axis=0)

    def add_patch(self, mask, key, init_clr, grad_clr):
        patch = PatchTF(mask/255, key, init_clr, grad_clr)
        self.patches[key] = patch
        return patch.mask_tf

    def add_image(self, img):
        image = ImageTF(img)
        self.images.append(image)

    def compile(self):
        for patch_key in self.patches:
            self.patches[patch_key].set_grid(self.images[0])
            self.patches[patch_key].init_src_parts()
            self.patches[patch_key].init_transforms(self.images)
            self.patches[patch_key].pack_transforms()

    def prepare_imgs(self):
        img_sh = self.images[0].get_image().shape
        imgs = np.zeros((len(self.images), img_sh[0], img_sh[1], img_sh[2]))
        for i in range(len(self.images)):
            imgs[i] = self.images[i].get_image()
        self.imgs_tf = tf.Variable(initial_value=imgs, dtype=tf.float32, name="input")

        sum_mask = tf.zeros_like(self.imgs_tf)
        with tf.variable_scope("prepare_img"):
            for patch_key in self.patches:
                for i, part in enumerate(self.patches[patch_key].src_masks_tf):
                    patch_padded = self.pad_img_tf(self.imgs_tf[0], part)
                    patch_batched = self.batchify(patch_padded)
                    Mb = self.patches[patch_key].batch_transform[i]
                    patch_transformed = tf.contrib.image.transform(patch_batched, Mb,
                                                            interpolation="NEAREST",
                                                            name="perspective")
                    if (self.patches[patch_key].mask_np.shape[2] == 1):
                        patch_transformed = tf.concat([patch_transformed for i in range(3)], axis=3)
                    sum_mask = tf.where(patch_transformed > 0.0, patch_transformed, sum_mask)
            inv_sum_mask = 1.0 - sum_mask
            self.stencil = inv_sum_mask
            img_mod = self.imgs_tf[...,::-1]/255.0 * inv_sum_mask
        return tf.assign(self.imgs_tf, img_mod)

    def apply_patches(self, colorizer):
        sum_mask = tf.zeros_like(self.imgs_tf)
        with tf.variable_scope("apply_patches"):
            for patch_key in self.patches:
                for i, part in enumerate(self.patches[patch_key].src_masks_tf):
                    patch = self.patches[patch_key].mask_tf
                    if (self.patches[patch_key].mask_np.shape[2] != 3):
                        patch = colorizer(patch)
                    part_patch = patch * part
                    patch_padded = self.pad_img_tf(self.imgs_tf[0], part_patch)
                    patch_batched = self.batchify(patch_padded)
                    Mb = self.patches[patch_key].batch_transform[i]
                    patch_transformed = tf.contrib.image.transform(patch_batched, Mb,
                                                            interpolation="NEAREST",
                                                            name="perspective")
                    sum_mask = sum_mask + patch_transformed
        return self.imgs_tf + sum_mask

    def init_vars(self):
        vars = [self.imgs_tf]
        for patch_key in self.patches:
            vars.append(self.patches[patch_key].mask_tf)
            for part in self.patches[patch_key].src_masks_tf:
                vars.append(part)
        return tf.initializers.variables(vars)