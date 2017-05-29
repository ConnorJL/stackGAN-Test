import os

import numpy as np
from scipy.misc import imsave
import tensorflow as tf
from tensorflow.contrib import losses
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops


slim = tf.contrib.slim


# User input
amount = 9


log_dir = "log"
batch_size = 4
z_dim = 100

if not os.path.exists("samples"):
    os.mkdir("samples")

def generator1(z, training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
    }

    c0 = tf.reshape(z, [batch_size, 1, 1, z_dim])

    with arg_scope(
        [slim.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):

        gen = tf.image.resize_nearest_neighbor(c0, [2,2])
        gen = slim.conv2d(gen, 1024, [3,3], [1,1], scope="convolution1")

        gen = tf.image.resize_nearest_neighbor(gen, [4,4])
        gen = slim.conv2d(gen, 512, [3,3], [1,1], scope="convolution2")

        gen = tf.image.resize_nearest_neighbor(gen, [8,8])
        gen = slim.conv2d(gen, 256, [3,3], [1,1], scope="convolution3")

        gen = tf.image.resize_nearest_neighbor(gen, [16,16])
        gen = slim.conv2d(gen, 128, [3,3], [1,1], scope="convolution4")

        gen = tf.image.resize_nearest_neighbor(gen, [32,32])
        gen = slim.conv2d(gen, 64, [3,3], [1,1], scope="convolution5")

        gen = tf.image.resize_nearest_neighbor(gen, [64,64])

        # l = [(4096, [3,3], [2,2]), (2048, [3,3], [2,2]), (2048, [3,3], [2,2]),
        #     (2048, [3,3], [2,2]), (1024, [3,3], [2,2]), (512, [3,3], [2,2]), (256, [3,3], [2,2]), (128, [3,3], [1,1]), (3, [3,3], [1,1])]


    gen = slim.conv2d(gen, 3, [3,3], [1,1], weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(), activation_fn=tf.tanh, scope="convolutionend")


    return gen


def generator2(s, training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
    }

    with arg_scope(
        [slim.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=lrelu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):

        s = slim.stack(s, slim.conv2d, [(64, [4,4], [2,2]), (128, [4,4], [2,2])], scope="convolution")

    # Res
    with arg_scope(
        [slim.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):

        res = slim.conv2d(s, 256, [3,3], [1,1], scope="res1_1")
        res = slim.conv2d(res, 256, [3,3], [1,1], scope="res1_2")
        gen = s + res

        res = slim.conv2d(gen, 256, [3,3], [1,1], scope="res2_1")
        res = slim.conv2d(res, 256, [3,3], [1,1], scope="res2_2")
        gen = gen + res

    with arg_scope(
        [slim.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):

        gen = tf.image.resize_nearest_neighbor(gen, [64,64])
        gen = slim.conv2d(gen, 1024, [3,3], [1,1], scope="convolution_2")

        gen = tf.image.resize_nearest_neighbor(gen, [128,128])
        gen = slim.conv2d(gen, 512, [3,3], [1,1], scope="convolution_2_1")

        # l = [(4096, [3,3], [2,2]), (2048, [3,3], [2,2]), (2048, [3,3], [2,2]),
        #     (2048, [3,3], [2,2]), (1024, [3,3], [2,2]), (512, [3,3], [2,2]), (256, [3,3], [2,2]), (128, [3,3], [1,1]), (3, [3,3], [1,1])]


    gen = slim.conv2d(gen, 3, [3,3], [1,1], weights_regularizer=regularizers.l2_regularizer(weight_decay),
    weights_initializer=initializers.variance_scaling_initializer(), activation_fn=tf.tanh, scope="convolutionend")


    return gen

def lrelu(x, leak=0.2):
     return tf.maximum(x, leak*x)


def color_grid_vis(X, nh, nw, save_path=None):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    if save_path is not None:
        imsave(save_path, img)
    return img


def main():

    z = tf.placeholder(tf.float32, [None, z_dim], name='z')

    with tf.variable_scope("generator1") as scope:
        gen1 = generator1(z, training=False)

    with tf.variable_scope("generator2") as scope:
        gen2 = generator2(gen1, training=False)

    g_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator1")
    g_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator2")


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Model found! Restoring...")
            variables_to_restore = g_vars1 + g_vars2
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, ckpt.model_checkpoint_path)
            print("Restored!")
        else:
            print("No model found!")
            return

        def visualize(image_amount=9):
            images1 = []
            images2 = []
            done1 = False
            done2 = False
            num = 0
            for root, dirs, files in os.walk("samples"):
                for f in files:
                    if num <= int(f.split("_")[-1].split("x")[0]):
                        num = int(f.split("_")[-1].split("x")[0])+1

            while not done1:
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
                img = sess.run(gen1, feed_dict={z: batch_z})
                img = np.squeeze(img)
                for i in xrange(batch_size):
                    images1.append(np.reshape(img[i], [64, 64, 3]))
                    if len(images1) == image_amount*image_amount:
                        color_grid_vis(images1, image_amount, image_amount, save_path=os.path.join("samples", "sample_" + str(num) + "x1" + ".png"))
                        done1 = True
                        break

            while not done2:
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
                img = sess.run(gen2, feed_dict={z: batch_z})
                img = np.squeeze(img)
                for i in xrange(batch_size):
                    images2.append(np.reshape(img[i], [128, 128, 3]))
                    if len(images2) == image_amount*image_amount:
                        color_grid_vis(images2, image_amount, image_amount, save_path=os.path.join("samples", "sample_" + str(num) + "x2" + ".png"))
                        done2 = True
                        break
        try:
            print("Starting sampling!")
            visualize(image_amount=amount)
        finally:
            coord.request_stop()

        coord.join(threads)

main()
