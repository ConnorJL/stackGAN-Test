# TODO: Multivariate normal distribution, Hyperparameters, better variable names
# TODO: Combine sampler 1+2 for visualize

import os
# import time

import numpy as np
from scipy.misc import imsave
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops

# Insert your data loader here! Expects a tensorflow queue or equivalent
from loader import get_dataset

slim = tf.contrib.slim

log_dir = "log"
imgs_dir = "imgs"
data_dir = ""
batch_size = 4
max_iterations = 100000000
sum_per = 5 # Create a summary every this many steps
save_per = 10000 # Save a check point every this many steps
learning_rate = 0.00005 # 0.0002
d_iters = 5 # Number of training steps the discriminator takes before the generator
z_dim = 100 # Dimension of the noise vector
start_2 = 140000 # 625000 ~20 epochs
WGAN = True # Whether or not to use WGAN weight clipping
c = 0.01 # If using WGAN, the value to clip weights within
image_per = 1000 # Create a sample image every this many steps


if not os.path.exists(log_dir):
    os.mkdir(log_dir)

if not os.path.exists(imgs_dir):
    os.mkdir(imgs_dir)


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

        s = slim.stack(s, slim.conv2d, [(64, [4,4], [2,2]), (128*2, [4,4], [2,2])], scope="convolution")

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



def discriminator1(img, training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
    }

    disc = slim.conv2d(img, 64*2, [4,4], [2,2], weights_regularizer=regularizers.l2_regularizer(weight_decay),
    weights_initializer=initializers.variance_scaling_initializer(), activation_fn=lrelu, scope="convolutionstart")

    with arg_scope(
        [slim.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=lrelu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):

        disc = slim.stack(disc, slim.conv2d, [(128, [4,4], [2,2]),
            (256, [4,4], [2,2]), (512, [4,4], [2,2])], scope="convolution")


    disc = slim.conv2d(disc, 512, [1,1], weights_regularizer=regularizers.l2_regularizer(weight_decay),
    weights_initializer=initializers.variance_scaling_initializer(), activation_fn=lrelu, scope="convolutionend")

    disc = tf.reshape(disc, [batch_size, 4*4*512])
    disc = slim.fully_connected(disc, 1, activation_fn=None, scope="logits")

    return disc


def discriminator2(img, training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
    }

    disc = slim.conv2d(img, 64, [4,4], [2,2], weights_regularizer=regularizers.l2_regularizer(weight_decay),
    weights_initializer=initializers.variance_scaling_initializer(), activation_fn=lrelu, scope="convolutionstart")
    with arg_scope(
        [slim.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=lrelu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):

        disc = slim.stack(disc, slim.conv2d, [(128, [4,4], [2,2]),
            (256, [4,4], [2,2]), (512, [4,4], [2,2]), (1024, [4,4], [2,2])], scope="convolution")


    disc = slim.conv2d(disc, 1024, [1,1], weights_regularizer=regularizers.l2_regularizer(weight_decay),
    weights_initializer=initializers.variance_scaling_initializer(), activation_fn=lrelu, scope="convolutionend")

    disc = tf.reshape(disc, [batch_size, 4*4*1024])
    disc = slim.fully_connected(disc, 1, activation_fn=None, scope="logits")

    return disc


def main():
    images = get_dataset(data_dir, batch_size)
    images1 = tf.image.resize_bilinear(images, [64, 64],
                                     align_corners=False)
    tf.image_summary("real", images, max_images=1)

    z = tf.placeholder(tf.float32, [None, z_dim], name='z')

    with tf.variable_scope("generator1") as scope:
        gen1 = generator1(z)
        tf.image_summary("fake1", gen1, max_images=1)
        scope.reuse_variables()
        sampler1 = generator1(z, training=False)
        tf.image_summary("fake1_sampler", sampler1, max_images=1)

    with tf.variable_scope("generator2") as scope:
        gen2 = generator2(gen1)
        tf.image_summary("fake2", gen2, max_images=1)
        scope.reuse_variables()
        sampler2 = generator2(gen1, training=False)
        tf.image_summary("fake2_sampler", sampler2, max_images=1)

    with tf.variable_scope("discriminator1") as scope:
        disc_real1 = discriminator1(images1)
        scope.reuse_variables()
        disc_fake1 = discriminator1(gen1)

    with tf.variable_scope("discriminator2") as scope:
        disc_real2 = discriminator2(images)
        scope.reuse_variables()
        disc_fake2 = discriminator2(gen2)


    d_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator1")
    g_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator1")
    d_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator2")
    g_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator2")

    disc_real_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real1, tf.ones(tf.shape(disc_real1))))
    disc_fake_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake1, tf.fill(tf.shape(disc_real1), -1.0)))

    disc_real_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real2, tf.ones(tf.shape(disc_real1))))
    disc_fake_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake2, tf.fill(tf.shape(disc_real1), -1.0)))

    d_loss1 = disc_real_loss1 + (disc_fake_loss1 / 2.0)
    d_loss2 = disc_real_loss2 + (disc_fake_loss2 / 2.0)

    g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake1, tf.ones(tf.shape(disc_real1))))
    g_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake2, tf.ones(tf.shape(disc_real1))))

    tf.scalar_summary("Discriminator_loss_real1", disc_real_loss1)
    tf.scalar_summary("Discrimintator_loss_fake1", disc_fake_loss1)
    tf.scalar_summary("Discriminator_loss1", d_loss1)
    tf.scalar_summary("Generator_loss1", g_loss1)
    tf.scalar_summary("Discriminator_loss_real2", disc_real_loss2)
    tf.scalar_summary("Discrimintator_loss_fake2", disc_fake_loss2)
    tf.scalar_summary("Discriminator_loss2", d_loss2)
    tf.scalar_summary("Generator_loss2", g_loss2)

    if WGAN:
        d_optimizer1 = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        g_optimizer1 = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        d_optimizer2 = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        g_optimizer2 = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    else:
        d_optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        g_optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        d_optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
        g_optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate)

    d_train_op1 = slim.learning.create_train_op(d_loss1, d_optimizer1, variables_to_train=d_vars1)
    g_train_op1 = slim.learning.create_train_op(g_loss1, g_optimizer1, variables_to_train=g_vars1)

    d_train_op2 = slim.learning.create_train_op(d_loss2, d_optimizer2, variables_to_train=d_vars2)
    g_train_op2 = slim.learning.create_train_op(g_loss2, g_optimizer2, variables_to_train=g_vars2)

    clip_critic1 = []
    for var in d_vars1:
        clip_critic1.append(tf.assign(var, tf.clip_by_value(var, -c, c)))

    clip_critic2 = []
    for var in d_vars2:
        clip_critic2.append(tf.assign(var, tf.clip_by_value(var, -c, c)))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)
        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        start = 0

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Model found! Restoring...")
            start = int(ckpt.model_checkpoint_path.split("-")[-1])+1
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Restored!")
        else:
            print("No model found!")

        def make_feed_dict():
            batch_z = np.random.uniform(-1., 1., [batch_size, z_dim]).astype(np.float32)
            feed = {z: batch_z}
            return feed

        def visualize(step, image_amount = 9):
            images1 = []
            images2 = []
            done1 = False
            done2 = False
            while not done1:
                img = sess.run(sampler1, feed_dict=make_feed_dict())
                img = np.squeeze(img)
                for i in xrange(batch_size):
                    images1.append(np.reshape(img[i], [64, 64, 3]))
                    if len(images1) == image_amount*image_amount:
                        color_grid_vis(images1, image_amount, image_amount, save_path=os.path.join(imgs_dir, "test_" + str(step) + "x1" + ".png"))
                        done1 = True
                        break

            while not done2:
                img = sess.run(sampler2, feed_dict=make_feed_dict())
                img = np.squeeze(img)
                for i in xrange(batch_size):
                    images2.append(np.reshape(img[i], [128, 128, 3]))
                    if len(images2) == image_amount*image_amount:
                        color_grid_vis(images2, image_amount, image_amount, save_path=os.path.join(imgs_dir, "test_" + str(step) + "x2" + ".png"))
                        done2 = True
                        break

        try:
            curr = start
            print("Starting training!")
            r1 = 1
            r2 = 1
            for itr in xrange(start, max_iterations):
                # start_time = time.time()
                if WGAN:
                    if itr < 25 or (start_2 < itr and itr < (start_2 + 25)) or itr % 500 == 0:
                        diters = 100
                    else:
                        diters = d_iters
                else:
                    if r1 < 0.1:
                        diters = 1
                    elif r1 > 10:
                        diters = 1
                    else:
                        diters = 1

                for i in xrange(diters):
                    if WGAN:
                        sess.run(clip_critic1)
                    sess.run(d_train_op1, feed_dict=make_feed_dict())

                sess.run(g_train_op1, feed_dict=make_feed_dict())

                if start_2 < itr:
                    if WGAN:
                        if itr < 25 or (start_2 < itr and itr < (start_2 + 25)) or itr % 500 == 0:
                            diters = 100
                        else:
                            diters = d_iters
                    else:
                        if r2 < 0.1:
                            diters = 1
                        elif r2 > 10:
                            diters = 1
                        else:
                            diters = 1

                    for i in xrange(diters):
                        if WGAN:
                            sess.run(clip_critic2)
                        sess.run(d_train_op2, feed_dict=make_feed_dict())

                    sess.run(g_train_op2, feed_dict=make_feed_dict())

                if itr % sum_per == 0:
                    g_loss_val, d_loss_val, g_loss_val2, d_loss_val2, summary_str = sess.run([g_loss1, d_loss1, g_loss2, d_loss2, summary_op], feed_dict=make_feed_dict())
                    print("Step: %d, generator1 loss: %g, discriminator1_loss: %g" % (itr, g_loss_val, d_loss_val))
                    print("Step: %d, generator2 loss: %g, discriminator2_loss: %g" % (itr, g_loss_val2, d_loss_val2))
                    if not WGAN:
                        r1 = max(g_loss_val, 0.000001) / max(d_loss_val, 0.000001)
                        r2 = max(g_loss_val2, 0.000001) / max(d_loss_val2, 0.000001)
                    summary_writer.add_summary(summary_str, itr)
                    # print("--- %s seconds ---" % (time.time() - start_time))


                if itr % save_per == 0:
                    saver.save(sess, os.path.join(log_dir, "model.ckpt"), global_step=itr)

                if itr % image_per == 0:
                    visualize(itr)

                curr = itr

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print("Ending Training...")
            saver.save(sess, os.path.join(log_dir, "model.ckpt"), global_step=curr)
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == "__main__":
    main()
