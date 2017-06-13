from __future__ import division
import os
import sys
import time
import math
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from six.moves import xrange
from PIL import Image

from ops import *

FLAGS = None

DATASET_NAME = 'mnist'
DATA_PATH = 'data/' + DATASET_NAME
CHECKPOINT_DIR = 'checkpoint/' + DATASET_NAME
LOG_DIR = 'log/' + DATASET_NAME
SAMPLE_DIR = 'samples/' + DATASET_NAME

BETA1 = 0.5
BETA2 = 0.9
LAMB_GP = 10

DATA_DIM = 28 * 28
IMAGE_DIM = 1
NOISE_DIM = 128


def train(sess):

    real_data_holder = tf.placeholder(
        tf.float32, [FLAGS.batch_size, DATA_DIM], name='real_data')
    input_noise_holder = tf.placeholder(
        tf.float32, [FLAGS.batch_size, NOISE_DIM], name='input_noise')

    fake_data = generator(input_noise_holder)
    real_score = discriminator(real_data_holder)
    fake_score = discriminator(fake_data, reuse=True)
    t_vars = tf.trainable_variables()
    gene_vars = [var for var in t_vars if 'generator' in var.name]
    disc_vars = [var for var in t_vars if 'discriminator' in var.name]
    sampler = generator(input_noise_holder, is_train=False)

    if not FLAGS.iwgan:
        all_score = tf.concat([real_score, fake_score], axis=0)
        labels_gene = tf.ones([FLAGS.batch_size])
        labels_disc = tf.concat(
            [tf.ones([FLAGS.batch_size]), tf.zeros([FLAGS.batch_size])], axis=0)
        gene_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_gene, logits=fake_score))
        disc_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_disc, logits=all_score))

        gene_train_op = tf.train.AdamOptimizer(
            FLAGS.learning_rate, BETA1).minimize(
                gene_loss, var_list=gene_vars)
        disc_train_op = tf.train.AdamOptimizer(
            FLAGS.learning_rate, BETA1).minimize(
                disc_loss, var_list=disc_vars)
    else:
        gene_loss = -tf.reduce_mean(fake_score)
        disc_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
        alpha = tf.random_uniform(
            shape=[FLAGS.batch_size, 1, 1, 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data_holder
        interpolates = real_data_holder + (alpha * differences)
        gradients = tf.gradients(discriminator(
            interpolates, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(
            tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
        disc_loss += LAMB_GP * gradient_penalty

        gene_train_op = tf.train.AdamOptimizer(
            FLAGS.learning_rate, BETA1, BETA2).minimize(gene_loss, var_list=gene_vars)
        disc_train_op = tf.train.AdamOptimizer(
            FLAGS.learning_rate, BETA1, BETA2).minimize(disc_loss, var_list=disc_vars)

    tf.summary.scalar('gene_loss', gene_loss)
    tf.summary.scalar('disc_loss', disc_loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    tf.global_variables_initializer().run()
    counter = 1
    if FLAGS.old_only:
        variables_to_restore = slim.get_variables_to_restore(
            include=['discriminator/hidden1/conv_old', 'discriminator/hidden1/bn_old',
                     'discriminator/hidden2/conv_old', 'discriminator/hidden2/bn_old',
                     'generator/hidden2/conv_old', 'generator/hidden2/bn_old',
                     'generator/hidden3/conv_old'])
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver()
    start_time = time.time()
    could_load, checkpoint_counter = load(sess, saver, CHECKPOINT_DIR)
    if could_load:
        counter = checkpoint_counter
        print ' [*] Load SUCCESS'
    else:
        print ' [!] Load failed...'

    """
    if FLAGS.is_test:
        index = 0
        file_object = open(SAMPLE_PATH, 'rb')
        for image_batch in read_in_chunks(file_object, FLAGS.batch_size):
            if image_batch.shape[0] != FLAGS.batch_size:
                break
            could_load, checkpoint_counter = load(sess, saver, CHECKPOINT_DIR)

            image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5

            masked_image_batch, hiding_image_batch, masks_idx = preprocess_image(
                image_batch)
            samples, gene_loss_value, disc_loss_value = sess.run(
                [sampler, gene_loss, disc_loss],
                feed_dict={
                    real_data_holder: masked_image_batch,
                    input_noise_holder: hiding_image_batch
                })
            inpaint_image = np.copy(masked_image_batch)
            for idx in range(FLAGS.batch_size):
                idx_start1 = int(masks_idx[idx, 0])
                idx_end1 = int(masks_idx[idx, 0] + (HIDING_SIZE))
                idx_start2 = int(masks_idx[idx, 1])
                idx_end2 = int(masks_idx[idx, 1] + (HIDING_SIZE))
                inpaint_image[idx, idx_start1: idx_end1,
                              idx_start2: idx_end2, :] = samples[idx, :, :, :]

            save_all_images(index, 0, image_batch)
            save_all_images(index, 1, masked_image_batch)
            save_all_images(index, 2, inpaint_image)
            print(
                '[Sample %2d] G_loss: %.8f, D_loss: %.8f'
                % (index, gene_loss_value, disc_loss_value / LAMB_ADV))
            index += 1
        return
    """

    fixed_noise = np.random.uniform(-1, 1,
                                    [FLAGS.batch_size, NOISE_DIM]).astype(np.float32)
    mnist = input_data.read_data_sets(DATA_PATH)
    for epoch in xrange(FLAGS.epoch):

        all_batch = mnist.train.next_batch(FLAGS.batch_size)
        data_batch = all_batch[0]
        # data_batch = (data_batch.astype(np.float32) - 127.5) / 127.5
        noise_batch = np.random.uniform(-1, 1,
                                        [FLAGS.batch_size, NOISE_DIM]).astype(np.float32)

        if epoch % FLAGS.sample == 0:
            summary, samples, gene_loss_value, disc_loss_value = sess.run(
                [merged, sampler, gene_loss, disc_loss],
                feed_dict={
                    real_data_holder: data_batch,
                    input_noise_holder: fixed_noise
                })
            writer.add_summary(summary, epoch)

            save_all_data(epoch, 0, np.reshape(samples, (-1, 28, 28, 1)))

            print(
                '[Getting Sample...] G_loss: %2.8f, D_loss: %2.8f'
                % (gene_loss_value, disc_loss_value))

        if epoch % FLAGS.ckpt == 0:
            save(sess, saver, CHECKPOINT_DIR, counter)

        if epoch % 3 == 0:
            _, gene_loss_value, disc_loss_value = sess.run(
                [disc_train_op, gene_loss, disc_loss],
                feed_dict={
                    real_data_holder: data_batch,
                    input_noise_holder: noise_batch
                })

        _, gene_loss_value, disc_loss_value = sess.run(
            [gene_train_op, gene_loss, disc_loss],
            feed_dict={
                real_data_holder: data_batch,
                input_noise_holder: noise_batch
            })

        print(
            'batch: %4d time: %4.2f, G_loss: %2.8f, D_loss: %2.8f'
            % (epoch, time.time() - start_time, gene_loss_value,
               disc_loss_value))
        counter += 1


def discriminator(data, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        layer_num = 1
        with tf.variable_scope('hidden' + str(layer_num)):
            hidden = conv2d(tf.reshape(data, [-1, 28, 28, 1]),
                            16, k_h=3, k_w=3, d_h=2, d_w=2, name='conv_old')
            hidden = lrelu(batch_norm(hidden, name='bn_old'))

        layer_num += 1
        with tf.variable_scope('hidden' + str(layer_num)):
            hidden = conv2d(hidden, 32, k_h=3, k_w=3,
                            d_h=2, d_w=2, name='conv_old')
            hidden = lrelu(batch_norm(hidden, name='bn_old'))

        layer_num += 1
        with tf.variable_scope('hidden' + str(layer_num)):
            hidden = linear(tf.reshape(
                hidden, [FLAGS.batch_size, -1]), 128, 'fc_new')
            hidden = lrelu(batch_norm(hidden, name='bn_new'))

        layer_num += 1
        with tf.variable_scope('hidden' + str(layer_num)):
            hidden = linear(hidden, 1, 'fc_new')

        return hidden[:, 0]


def generator(noise, is_train=True):
    with tf.variable_scope('generator') as scope:
        if not is_train:
            scope.reuse_variables()

        layer_num = 1
        with tf.variable_scope('hidden' + str(layer_num)):
            hidden = linear(noise, 7 * 7 * 32, 'fc_new')
            hidden = tf.nn.relu(batch_norm(
                hidden, train=is_train, name='bn_new'))
            hidden = tf.reshape(hidden, [-1, 7, 7, 32])

        layer_num += 1
        with tf.variable_scope('hidden' + str(layer_num)):
            hidden = deconv2d(hidden, [FLAGS.batch_size, 14, 14, 16],
                              k_h=3, k_w=3, d_h=2, d_w=2, name='conv_old')
            hidden = tf.nn.relu(batch_norm(
                hidden, train=is_train, name='bn_old'))

        layer_num += 1
        with tf.variable_scope('hidden' + str(layer_num)):
            hidden = deconv2d(hidden, [FLAGS.batch_size, 28, 28, 1],
                              k_h=3, k_w=3, d_h=2, d_w=2, name='conv_old')
            hidden = tf.nn.sigmoid(hidden)

        return tf.reshape(hidden, [-1, DATA_DIM])


"""
def discriminator(data, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        hidden = conv2d(tf.reshape(data, [-1, 28, 28, 1]),
                        64, k_h=5, k_w=5, d_h=2, d_w=2, name='d_h1_conv')
        hidden = lrelu(batch_norm(hidden, name='d_h1_bn'))

        hidden = conv2d(hidden, 128, k_h=5, k_w=5,
                        d_h=2, d_w=2, name='d_h2_conv')
        hidden = lrelu(batch_norm(hidden, name='d_h2_bn'))

        hidden = conv2d(hidden, 256, k_h=5, k_w=5,
                        d_h=2, d_w=2, name='d_h3_conv')
        hidden = lrelu(batch_norm(hidden, name='d_h3_bn'))

        output = linear(tf.reshape(
            hidden, [FLAGS.batch_size, -1]), 1, 'd_h4_fc')

        return output[:, 0]


def generator(noise, is_train=True):
    with tf.variable_scope('generator') as scope:
        if not is_train:
            scope.reuse_variables()

        hidden = linear(noise, 4 * 4 * 256, 'g_h1_fc')
        hidden = tf.nn.relu(batch_norm(hidden, train=is_train, name='g_h1_bn'))
        hidden = tf.reshape(hidden, [-1, 4, 4, 256])

        hidden = deconv2d(hidden, [FLAGS.batch_size, 8, 8, 16],
                          k_h=5, k_w=5, d_h=2, d_w=2, name='g_h2_conv')
        hidden = tf.nn.relu(batch_norm(hidden, train=is_train, name='g_h2_bn'))

        hidden = hidden[:, :7, :7, :]

        hidden = deconv2d(hidden, [FLAGS.batch_size, 14, 14, 64],
                          k_h=3, k_w=3, d_h=2, d_w=2, name='g_h3_conv')
        hidden = tf.nn.relu(batch_norm(hidden, train=is_train, name='g_h3_bn'))

        hidden = deconv2d(hidden, [FLAGS.batch_size, 28, 28, 1],
                          k_h=5, k_w=5, d_h=2, d_w=2, name='g_h4_conv')
        output = tf.nn.sigmoid(hidden)

        return tf.reshape(output, [-1, DATA_DIM])
"""


def save(sess, saver, checkpoint_dir, step):
    model_name = "model"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(
        sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def load(sess, saver, checkpoint_dir):
    import re
    print ' [*] Reading checkpoints...'

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print " [*] Success to read {}".format(ckpt_name)
        return True, counter
    else:
        print ' [*] Failed to find a checkpoint'
        return False, 0


def read_in_chunks(file_object, chunk_size):
    while True:
        size = NOISE_DIM
        batch = np.fromfile(
            file_object, dtype=np.uint8, count=size * chunk_size)
        if batch is None:
            break
        data = np.reshape(batch, (-1, NOISE_DIM))
        yield data


def combine_images(images):
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = images.shape[1:4]
    output_image = np.zeros(
        (height * shape[0], width * shape[1], shape[2]),
        dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        output_image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = \
            img[:, :, :]
    return output_image


def save_all_data(epoch, index, input_image):
    image = combine_images(input_image)
    # image = image * 127.5 + 127.5
    image = image * 255.99

    if FLAGS.is_test:
        image_path = SAMPLE_DIR + "_test/" + \
            str(epoch) + "_" + str(index) + ".png"
    else:
        image_path = SAMPLE_DIR + "/" + str(epoch) + "_" + str(index) + ".png"
    if IMAGE_DIM == 1:
        image = np.squeeze(image)
        Image.fromarray(image.astype(np.uint8), mode='L').save(image_path)
    else:
        Image.fromarray(image.astype(np.uint8)).save(image_path)


def main(_):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    run_config = tf.ConfigProto(allow_soft_placement=True)
    run_config.gpu_options.allow_growth = True
    print 'Using GPU ' + str(FLAGS.gpu)
    with tf.device('/gpu:' + str(FLAGS.gpu)):
        with tf.Session(config=run_config) as sess:
            train(sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0002,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=20000,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--is_test',
        type=bool,
        default=False,
        help='Test or not.'
    )
    parser.add_argument(
        '--iwgan',
        type=bool,
        default=False,
        help='Using improved wgan or not.'
    )
    parser.add_argument(
        '--old_only',
        type=bool,
        default=False,
        help='Saving new variables or not.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size.'
    )
    parser.add_argument(
        '--ckpt',
        type=int,
        default=100,
        help='Save checkpoint every ? epochs.'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=100,
        help='Get sample every ? epochs.'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=2,
        help='GPU No.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
