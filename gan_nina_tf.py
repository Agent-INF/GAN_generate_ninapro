from __future__ import division
import os
#import sys
import time
import math
#import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.io as sio
#from PIL import Image

from ops import *

FLAG = tf.app.flags
FLAGS = FLAG.FLAGS

FLAG.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate.')
FLAG.DEFINE_integer('start_epoch', 0, 'Number of epoch to run at the start.')
FLAG.DEFINE_integer('epoch', 1000, 'Number of epochs the trainer will run.')
FLAG.DEFINE_boolean('is_test', False, 'Test or not.')
FLAG.DEFINE_boolean('iwgan', True, 'Using improved wgan or not.')
FLAG.DEFINE_boolean('old_param', False, 'Saving new variables or not.')
FLAG.DEFINE_boolean('diff_lr', False, 'If True, using different learning rate.')
FLAG.DEFINE_string('dataname', '001', 'The dataset name')
FLAG.DEFINE_float(
    'old_lr', 0, 'When diff_lr is true, old param will use this learning_rate.')
FLAG.DEFINE_integer('batch_size', 1024, 'Batch size.')
FLAG.DEFINE_integer('ckpt', 100, 'Save checkpoint every ? epochs.')
FLAG.DEFINE_integer('log', 10, 'Get sample every ? epochs.')
FLAG.DEFINE_integer('gene_iter', 1,
                    'Train generator how many times every batch.')
FLAG.DEFINE_integer('disc_iter', 10,
                    'Train discriminator how many times every batch.')
FLAG.DEFINE_integer('gpu', 2, 'GPU No.')

DATA_PATH = 'data/nina/' + FLAGS.dataname + '.tfrecords'
CHECKPOINT_DIR = 'checkpoint/dev_tf' + FLAGS.dataname
OLD_CHECKPOINT_DIR = 'checkpoint/mnist'
LOG_DIR = 'log/dev_tf' + FLAGS.dataname
SAMPLE_DIR = 'samples/dev_tf' + FLAGS.dataname

BETA1 = 0.5
BETA2 = 0.9
LAMB_GP = 10

DATA_DIM = 10
DATA_FRAME = 20
NOISE_DIM = 64

HIDDEN_FRAME = int(math.ceil(DATA_FRAME / 4))
HIDDEN_DIM = int(math.ceil(DATA_DIM / 4))

HIDDEN_HEIGHT = np.array([HIDDEN_FRAME, HIDDEN_FRAME * 2, HIDDEN_FRAME * 4])
HIDDEN_WIDTH = np.array([HIDDEN_DIM, HIDDEN_DIM * 2, HIDDEN_DIM * 4])

H_CROP_S = int((HIDDEN_HEIGHT[2] - DATA_FRAME) / 2)
H_CROP_E = int(H_CROP_S + DATA_FRAME)
W_CROP_S = int((HIDDEN_WIDTH[2] - DATA_DIM) / 2)
W_CROP_E = int(W_CROP_S + DATA_DIM)


def train(sess):

  real_data_holder, _ = read_data(DATA_PATH, FLAGS.epoch, FLAGS.batch_size)
  input_noise_holder = tf.placeholder(
      tf.float32, [FLAGS.batch_size, NOISE_DIM], name='input_noise')

  fake_data = generator(input_noise_holder)
  real_score = discriminator(real_data_holder)
  fake_score = discriminator(fake_data, reuse=True)
  sampler = generator(input_noise_holder, is_train=False)

  if not FLAGS.is_test:
    tf.summary.histogram('real_data', real_data_holder)
    tf.summary.image('real_data', real_data_holder, max_outputs=10)
    tf.summary.histogram('input_noise', input_noise_holder)
    tf.summary.histogram('samples', sampler)
    tf.summary.image('samples', sampler, max_outputs=10)

  all_vars = tf.trainable_variables()
  if FLAGS.diff_lr:
    new_vars = [var for var in all_vars if '_new' in var.name]
    old_vars = [var for var in all_vars if '_old' in var.name]
    new_gene_vars = [var for var in new_vars if 'generator' in var.name]
    new_disc_vars = [var for var in new_vars if 'discriminator' in var.name]
    old_gene_vars = [var for var in old_vars if 'generator' in var.name]
    old_disc_vars = [var for var in old_vars if 'discriminator' in var.name]
  else:
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    disc_vars = [var for var in all_vars if 'discriminator' in var.name]

  if FLAGS.iwgan:
    gene_loss = -tf.reduce_mean(fake_score)
    disc_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
    alpha = tf.random_uniform(
        shape=[FLAGS.batch_size, 1, 1, 1], minval=0., maxval=1.)
    differences = fake_data - real_data_holder
    interpolates = real_data_holder + (alpha * differences)
    gradients = tf.gradients(
        discriminator(interpolates, reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    disc_loss += LAMB_GP * gradient_penalty
  else:
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

  if FLAGS.is_test:
    pass
  elif FLAGS.diff_lr:
    new_opt = tf.train.AdamOptimizer(FLAGS.learning_rate, BETA1)
    old_opt = tf.train.AdamOptimizer(FLAGS.old_lr, BETA1)

    gene_grads = tf.gradients(gene_loss, new_gene_vars + old_gene_vars)
    new_gene_grads = gene_grads[:len(new_gene_vars)]
    old_gene_grads = gene_grads[len(new_gene_vars):]
    gene_train_op_new = new_opt.apply_gradients(
        zip(new_gene_grads, new_gene_vars))
    gene_train_op_old = old_opt.apply_gradients(
        zip(old_gene_grads, old_gene_vars))
    gene_train_op = tf.group(gene_train_op_new, gene_train_op_old)

    disc_grads = tf.gradients(disc_loss, new_disc_vars + old_disc_vars)
    new_disc_grads = disc_grads[:len(new_disc_vars)]
    old_disc_grads = disc_grads[len(new_disc_vars):]
    disc_train_op_new = new_opt.apply_gradients(
        zip(new_disc_grads, new_disc_vars))
    disc_train_op_old = old_opt.apply_gradients(
        zip(old_disc_grads, old_disc_vars))
    disc_train_op = tf.group(disc_train_op_new, disc_train_op_old)
  else:
    gene_train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, BETA1).minimize(
        gene_loss, var_list=gene_vars)
    disc_train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, BETA1).minimize(
        disc_loss, var_list=disc_vars)

  if not FLAGS.is_test:
    tf.summary.scalar('gene_loss', gene_loss)
    tf.summary.scalar('disc_loss', disc_loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  sess.run(init_op)

  counter = 1
  saver = tf.train.Saver()
  if FLAGS.old_param:
    variables_to_restore = slim.get_variables_to_restore(
        include=[
            'discriminator/hidden1/conv_old', 'discriminator/hidden1/bn_old',
            'discriminator/hidden2/conv_old', 'discriminator/hidden2/bn_old',
            'generator/hidden2/conv_old', 'generator/hidden2/bn_old',
            'generator/hidden3/conv_old'
        ])
    restore_saver = tf.train.Saver(variables_to_restore)
    could_load, checkpoint_counter = load(sess, restore_saver,
                                          OLD_CHECKPOINT_DIR)
  else:
    could_load, checkpoint_counter = load(sess, saver, CHECKPOINT_DIR)
  if could_load:
    counter = checkpoint_counter
    print ' [*] Load SUCCESS'
  else:
    print ' [!] Load failed...'

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  if FLAGS.is_test:
    try:
      index = 0
      while not coord.should_stop():
        noise_batch = np.random.uniform(-1, 1, [FLAGS.batch_size,
                                                NOISE_DIM]).astype(np.float32)
        samples, gene_loss_value, disc_loss_value = sess.run(
            [sampler, gene_loss, disc_loss],
            feed_dict={input_noise_holder: noise_batch})
        data = np.squeeze(samples)
        data = np.reshape(data, (FLAGS.batch_size * DATA_FRAME, DATA_DIM))
        label = int(FLAGS.dataname)
        repetition = index
        shape = np.array(data.shape)
        subject = 0
        matpath = SAMPLE_DIR + \
            '/000_%s_%03d.mat' % (FLAGS.dataname, repetition)
        print label, repetition, shape, subject, matpath
        sio.savemat(matpath, {
            'data': data,
            'label': label,
            'repetition': repetition,
            'shape': shape,
            'subject': subject
        })
        print('[Sample %2d] G_loss: %.8f, D_loss: %.8f' %
              (index, gene_loss_value, disc_loss_value))

        index += 1
        if index >= FLAGS.epoch:
          break
    except tf.errors.OutOfRangeError:
      print 'Done training for %d epochs, %d steps.' % (FLAGS.epoch, counter)
    finally:
      coord.request_stop()
  else:
    try:
      while not coord.should_stop():
        noise_batch = np.random.uniform(-1, 1, [FLAGS.batch_size,
                                                NOISE_DIM]).astype(np.float32)
        for _ in xrange(FLAGS.disc_iter):
          _, gene_loss_value, disc_loss_value = sess.run(
              [disc_train_op, gene_loss, disc_loss],
              feed_dict={input_noise_holder: noise_batch})
        for _ in xrange(FLAGS.gene_iter):
          _, gene_loss_value, disc_loss_value = sess.run(
              [gene_train_op, gene_loss, disc_loss],
              feed_dict={input_noise_holder: noise_batch})
        if (counter - 1) % FLAGS.log == 0:
          print('step: %4d, G_loss: %2.8f, D_loss: %2.8f' %
                (counter - 1, gene_loss_value, disc_loss_value))
          summary, samples = sess.run(
              [merged, sampler], feed_dict={input_noise_holder: noise_batch})
          writer.add_summary(summary, counter)

        if (counter - 1) % FLAGS.ckpt == 0:
          save(sess, saver, CHECKPOINT_DIR, counter)

        counter += 1
    except tf.errors.OutOfRangeError:
      print 'Done training for %d epochs, %d steps.' % (FLAGS.epoch, counter)
    finally:
      coord.request_stop()

  coord.join(threads)
  sess.close()


def discriminator(data, reuse=False):
  with tf.variable_scope('discriminator') as scope:
    if reuse:
      scope.reuse_variables()

    layer_num = 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(data, 16, k_h=3, k_w=3, d_h=2, d_w=2, name='conv_old')
      hidden = lrelu(batch_norm(hidden, name='bn_old'))

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 32, k_h=3, k_w=3, d_h=2, d_w=2, name='conv_old')
      hidden = lrelu(batch_norm(hidden, name='bn_old'))

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = linear(tf.reshape(hidden, [FLAGS.batch_size, -1]), 1, 'fc_new')

    return hidden[:, 0]


def generator(noise, is_train=True):
  with tf.variable_scope('generator') as scope:
    if not is_train:
      scope.reuse_variables()

    layer_num = 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = linear(noise, HIDDEN_HEIGHT[0] * HIDDEN_WIDTH[0] * 32, 'fc_new')
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train, name='bn_new'))
      hidden = tf.reshape(hidden, [-1, HIDDEN_HEIGHT[0], HIDDEN_WIDTH[0], 32])

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = deconv2d(
          hidden, [FLAGS.batch_size, HIDDEN_HEIGHT[1], HIDDEN_WIDTH[1], 16],
          k_h=3,
          k_w=3,
          d_h=2,
          d_w=2,
          name='conv_old')
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train, name='bn_old'))

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = deconv2d(
          hidden, [FLAGS.batch_size, HIDDEN_HEIGHT[2], HIDDEN_WIDTH[2], 1],
          k_h=3,
          k_w=3,
          d_h=2,
          d_w=2,
          name='conv_old')
      #hidden = tf.nn.sigmoid(hidden)

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = tf.maximum(0.0024, hidden)
      #hidden = tf.reshape(hidden, [-1, 12])

    return hidden[:, H_CROP_S:H_CROP_E, W_CROP_S:W_CROP_E]


def save(sess, saver, checkpoint_dir, step):
  model_name = "model"

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


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


def read_data(filename, epochs, batch_size):
  filename_queue = tf.train.string_input_producer([filename], num_epochs=epochs)

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'data_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'repetition': tf.FixedLenFeature([], tf.int64),
          'subject': tf.FixedLenFeature([], tf.int64)
      })
  data = tf.decode_raw(features['data_raw'], tf.float64)
  data.set_shape([DATA_FRAME * DATA_DIM])
  data = tf.reshape(tf.cast(data, tf.float32), [DATA_FRAME, DATA_DIM, 1])
  label = features['label']

  datas, sparse_labels = tf.train.shuffle_batch(
      [data, label],
      batch_size,
      num_threads=2,
      capacity=1000 + 3 * batch_size,
      min_after_dequeue=1000)

  return datas, sparse_labels


def combine_images(images):
  num = images.shape[0]
  width = int(math.sqrt(num))
  height = int(math.ceil(float(num) / width))
  shape = images.shape[1:4]
  output_image = np.zeros(
      (height * shape[0], width * shape[1], shape[2]), dtype=images.dtype)
  for index, img in enumerate(images):
    i = int(index / width)
    j = index % width
    output_image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = \
        img[:, :, :]
  return output_image

def main(_):

  print 'dataname is:      ' + str(FLAGS.dataname)
  print 'learning_rate is: ' + str(FLAGS.learning_rate)
  print 'epoch is:         ' + str(FLAGS.epoch)
  print 'start_epoch is:   ' + str(FLAGS.start_epoch)
  print 'is_test is:       ' + str(FLAGS.is_test)
  print 'iwgan is:         ' + str(FLAGS.iwgan)
  print 'old_param is:     ' + str(FLAGS.old_param)
  print 'diff_lr is:       ' + str(FLAGS.diff_lr)
  print 'old_lr is:        ' + str(FLAGS.old_lr)
  print 'batch_size is:    ' + str(FLAGS.batch_size)
  print 'ckpt is:          ' + str(FLAGS.ckpt)
  print 'log is:           ' + str(FLAGS.log)
  print 'gpu is:           ' + str(FLAGS.gpu)

  if not os.path.exists(DATA_PATH):
    print 'DATA_PATH not exists!'
    return
  if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
  if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  if not os.path.exists(SAMPLE_DIR):
    os.makedirs(SAMPLE_DIR)

  run_config = tf.ConfigProto(allow_soft_placement=True)
  run_config.gpu_options.allow_growth = True
  with tf.device('/gpu:' + str(FLAGS.gpu)):
    with tf.Session(config=run_config) as sess:
      train(sess)


if __name__ == '__main__':
  tf.app.run()
