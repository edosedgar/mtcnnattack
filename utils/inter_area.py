from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np

def inter_area_batch(im_inp,h,w,hs,ws):
	# Do INTER_AREA resize here
	# h, w - input size
	# hs, ws - scaled size
	whole = im_inp
	return tf.clip_by_value(whole,0.,1.)
 
def resize_area_batch(imgs, hs, ws):
    _, h, w, _ = imgs.shape
    with tf.variable_scope("resize_area"):
    	out = inter_area_batch(imgs, int(h), int(w), hs, ws)
    return out