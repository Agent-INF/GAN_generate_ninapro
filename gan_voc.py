from __future__ import division
import os
import sys
import time
import math
import argparse
import tensorflow as tf
import numpy as np
from six.moves import xrange
from PIL import Image

from ops import *

FLAGS = None

DATASET_NAME = 'voc128'
DATA_PATH = 'data/' + DATASET_NAME + '.bin'
CHECKPOINT_DIR = 'checkpoint/' + DATASET_NAME + '_gan'
SAMPLE_DIR = 'samples/' + DATASET_NAME + '_gan'

BETA1 = 0.5
BETA2 = 0.9
LAMB_GP = 10.

IMAGE_SIZE = int(128)
IMAGE_DIM = int(3)
NOISE_DIM = int(512)
HID_DIM = int(64)


def train(sess):

    real_data_holder = tf.placeholder(
        tf.float32, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM], name='real_data')
    input_noise_holder = tf.placeholder(
        tf.float32, [FLAGS.batch_size, NOISE_DIM], name='input_noise')

    fake_data = generator(input_noise_holder)
    real_score = discriminator(real_data_holder)
    fake_score = discriminator(fake_data, reuse=True)

    sampler = generator(input_noise_holder, is_train=False)

    t_vars = tf.trainable_variables()
    gene_vars = [var for var in t_vars if 'g_' in var.name]
    disc_vars = [var for var in t_vars if 'd_' in var.name]

    if not FLAGS.iwgan:
        all_score = tf.concat([real_score, fake_score], axis=0)
        labels_disc = tf.concat(
            [tf.ones([FLAGS.batch_size]), tf.zeros([FLAGS.batch_size])], axis=0)
        labels_gene = tf.ones([FLAGS.batch_size])
        disc_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_disc, logits=all_score))
        gene_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_gene, logits=fake_score))

        gene_optimizer = tf.train.AdamOptimizer(
            FLAGS.learning_rate * 10, BETA1)
        gene_vars_grads = gene_optimizer.compute_gradients(
            gene_loss, gene_vars)
        gene_vars_grads = map(lambda gv: gv if gv[0] is None else [
            tf.clip_by_value(gv[0], -10., 10.), gv[1]], gene_vars_grads)
        gene_train_op = gene_optimizer.apply_gradients(gene_vars_grads)

        disc_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, BETA1)
        disc_vars_grads = disc_optimizer.compute_gradients(
            disc_loss, disc_vars)
        disc_vars_grads = map(lambda gv: gv if gv[0] is None else [
            tf.clip_by_value(gv[0], -10., 10.), gv[1]], disc_vars_grads)
        disc_train_op = disc_optimizer.apply_gradients(disc_vars_grads)
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

    tf.global_variables_initializer().run()

    counter = 1
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
    for epoch in xrange(FLAGS.epoch):
        index = 0
        file_object = open(DATA_PATH, 'rb')
        print 'Current Epoch is: ' + str(epoch)

        for data_batch in read_in_chunks(file_object, FLAGS.batch_size):
            if data_batch.shape[0] != FLAGS.batch_size:
                break
            data_batch = (data_batch.astype(np.float32) - 127.5) / 127.5
            noise_batch = np.random.uniform(-1, 1,
                                            [FLAGS.batch_size, NOISE_DIM]).astype(np.float32)

            if epoch % FLAGS.sample == 0 and index == 0:
                samples, gene_loss_value, disc_loss_value = sess.run(
                    [sampler, gene_loss, disc_loss],
                    feed_dict={
                        real_data_holder: data_batch,
                        input_noise_holder: fixed_noise
                    })
                print(
                    '[Getting Sample...] G_loss: %2.8f, D_loss: %2.8f'
                    % (gene_loss_value, disc_loss_value))
                save_all_images(epoch, index, samples)

            if epoch % FLAGS.ckpt == 0 and index == 0:
                save(sess, saver, CHECKPOINT_DIR, counter)

            for _ in xrange(FLAGS.disc):
                _, gene_loss_value, disc_loss_value = sess.run(
                    [disc_train_op, gene_loss, disc_loss],
                    feed_dict={
                        real_data_holder: data_batch,
                        input_noise_holder: noise_batch
                    })

            for _ in xrange(FLAGS.gene):
                _, gene_loss_value, disc_loss_value = sess.run(
                    [gene_train_op, gene_loss, disc_loss],
                    feed_dict={
                        real_data_holder: data_batch,
                        input_noise_holder: noise_batch
                    })
            print(
                'Epoch: %3d batch: %4d time: %4.2f, G_loss: %2.8f, D_loss: %2.8f'
                % (epoch, index, time.time() - start_time, gene_loss_value,
                   disc_loss_value))

            index += 1
            counter += 1


def discriminator(data, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        output = conv2d(data, HID_DIM, k_h=5, k_w=5, name='d_h1_conv')
        output = lrelu(batch_norm(output, name='d_h1_bn'))

        output = conv2d(output, HID_DIM * 2, k_h=5, k_w=5, name='d_h2_conv')
        output = lrelu(batch_norm(output, name='d_h2_bn'))

        output = conv2d(output, HID_DIM * 4, k_h=5, k_w=5, name='d_h3_conv')
        output = lrelu(batch_norm(output, name='d_h3_bn'))

        output = conv2d(output, HID_DIM * 8, k_h=5, k_w=5, name='d_h4_conv')
        output = lrelu(batch_norm(output, name='d_h4_bn'))

        output = linear(tf.reshape(
            output, [FLAGS.batch_size, -1]), 1, 'd_h5_fc')

        return output[:, 0]


def generator(noise, is_train=True):
    with tf.variable_scope('generator') as scope:
        if not is_train:
            scope.reuse_variables()

        size_16 = int(IMAGE_SIZE / 16)
        size_8 = int(IMAGE_SIZE / 8)
        size_4 = int(IMAGE_SIZE / 4)
        size_2 = int(IMAGE_SIZE / 2)

        output = linear(noise, size_16 * size_16 * 8 * HID_DIM, 'g_h1_fc')
        output = tf.nn.relu(batch_norm(output, train=is_train, name='g_h1_bn'))
        output = tf.reshape(
            output, [FLAGS.batch_size, size_16, size_16, HID_DIM * 8])

        output = deconv2d(output, [FLAGS.batch_size, size_8,
                                   size_8, HID_DIM * 4], k_h=5, k_w=5, name='g_h2_deconv')
        output = tf.nn.relu(batch_norm(output, train=is_train, name='g_h2_bn'))

        output = deconv2d(output, [FLAGS.batch_size, size_4,
                                   size_4, HID_DIM * 2], k_h=5, k_w=5, name='g_h3_deconv')
        output = tf.nn.relu(batch_norm(output, train=is_train, name='g_h3_bn'))

        output = deconv2d(output, [FLAGS.batch_size, size_2,
                                   size_2, HID_DIM], k_h=5, k_w=5, name='g_h4_deconv')
        output = tf.nn.relu(batch_norm(output, train=is_train, name='g_h4_bn'))

        output = deconv2d(output, [FLAGS.batch_size, IMAGE_SIZE,
                                   IMAGE_SIZE, IMAGE_DIM], k_h=5, k_w=5, name='g_h5_deconv')
        output = tf.nn.tanh(output)

        return output


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
        size = IMAGE_SIZE * IMAGE_SIZE * IMAGE_DIM
        batch = np.fromfile(
            file_object, dtype=np.uint8, count=size * chunk_size)
        if batch is None:
            break
        images = np.reshape(batch, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM))
        yield images


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


def save_all_images(epoch, index, input_image):
    image = combine_images(input_image)
    image = image * 127.5 + 127.5
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
    if not os.path.exists(SAMPLE_DIR + "_test"):
        os.makedirs(SAMPLE_DIR + "_test")

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
        default=1000,
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
        '--batch_size',
        type=int,
        default=64,
        help='Batch size.'
    )
    parser.add_argument(
        '--ckpt',
        type=int,
        default=1,
        help='Save checkpoint every ? epochs.'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=1,
        help='Get sample every ? epochs.'
    )
    parser.add_argument(
        '--disc',
        type=int,
        default=1,
        help='Train D how many times every iter.'
    )
    parser.add_argument(
        '--gene',
        type=int,
        default=1,
        help='Train G how many times every iter.'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=2,
        help='GPU No.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
