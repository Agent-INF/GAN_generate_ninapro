from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from models import generator
from models import discriminator
from utils import save
from utils import load
from utils import read_by_batch

# pylint: disable=invalid-name

FLAG = tf.app.flags
FLAGS = FLAG.FLAGS
FLAG.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate.')
FLAG.DEFINE_boolean('fresh_start', False,
                    'Fresh start will delele all logs and checkpoints')
FLAG.DEFINE_integer('start_epoch', 0, 'Number of epoch to run at the start.')
FLAG.DEFINE_integer('epoch', 1000, 'Number of epochs the trainer will run.')
FLAG.DEFINE_boolean('is_test', False, 'Test or not.')
FLAG.DEFINE_boolean('iwgan', True, 'Using improved wgan or not.')
FLAG.DEFINE_boolean('old_param', False, 'Saving new variables or not.')
FLAG.DEFINE_boolean(
    'diff_lr', False, 'If True, using different learning rate.')
FLAG.DEFINE_string('dataname', '001', 'The dataset name')
FLAG.DEFINE_float(
    'old_lr', 0, 'When diff_lr is true, old param will use this learning_rate.')
FLAG.DEFINE_integer('batch_size', 1024, 'Batch size.')
FLAG.DEFINE_integer('ckpt', 1, 'Save checkpoint every ? epochs.')
FLAG.DEFINE_integer('sample', 1, 'Get sample every ? epochs.')
FLAG.DEFINE_integer('gene_iter', 1,
                    'Train generator how many times every batch.')
FLAG.DEFINE_integer('disc_iter', 1,
                    'Train discriminator how many times every batch.')
FLAG.DEFINE_integer('gpu', 0, 'GPU No.')

DATA_PATH = 'data/' + FLAGS.dataname + '.bin'
TEST_PATH = 'data/' + FLAGS.dataname + '_test.bin'
CHECKPOINT_DIR = 'checkpoint/' + FLAGS.dataname
OLD_CHECKPOINT_DIR = 'checkpoint/mnist'
LOG_DIR = 'log/' + FLAGS.dataname
SAMPLE_DIR = 'samples/' + FLAGS.dataname

BETA1 = 0.5
BETA2 = 0.9
LAMB_GP = 10

DATA_DIM = 10
DATA_FRAME = 1
NOISE_DIM = 3

HIDDEN_FRAME = int(math.ceil(DATA_FRAME / 4))
HIDDEN_DIM = int(math.ceil(DATA_DIM / 4))
HIDDEN_SHAPE = np.array([[HIDDEN_FRAME, HIDDEN_FRAME * 2, HIDDEN_FRAME * 4],
                         [HIDDEN_DIM, HIDDEN_DIM * 2, HIDDEN_DIM * 4]])
CROP = np.array([[int((HIDDEN_SHAPE[0][2] - DATA_FRAME) / 2),
                  int((HIDDEN_SHAPE[0][2] + DATA_FRAME) / 2)],
                 [int((HIDDEN_SHAPE[1][2] - DATA_DIM) / 2),
                  int((HIDDEN_SHAPE[1][2] + DATA_DIM) / 2)]])


def run_model(sess):

  real_data_holder = tf.placeholder(
      tf.float32, [FLAGS.batch_size, DATA_FRAME, DATA_DIM, 1], name='real_data')
  input_noise_holder = tf.placeholder(
      tf.float32, [FLAGS.batch_size, NOISE_DIM], name='input_noise')

  fake_data = generator(
      input_noise_holder, FLAGS.batch_size, HIDDEN_SHAPE, CROP)
  real_score = discriminator(real_data_holder, FLAGS.batch_size)
  fake_score = discriminator(fake_data, FLAGS.batch_size, reuse=True)
  sampler = generator(input_noise_holder, FLAGS.batch_size,
                      HIDDEN_SHAPE, CROP, is_train=False)

  if not FLAGS.is_test:
    tf.summary.histogram('real_data', real_data_holder)
    tf.summary.image('real_data', real_data_holder, max_outputs=10)
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
        discriminator(interpolates, FLAGS.batch_size, reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(
        tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    disc_loss += LAMB_GP * gradient_penalty
    correct_real = tf.greater_equal(real_score, 0)
    correct_fake = tf.less(fake_score, 0)
    if FLAGS.is_test:
      disc_acc = tf.reduce_mean(tf.cast(correct_real, tf.float32))
      #disc_acc = tf.reduce_mean(tf.cast(correct_fake, tf.float32))
    else:
      disc_acc = tf.reduce_mean(
          tf.cast(tf.concat([correct_real, correct_fake], axis=0), tf.float32))
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
      correct = tf.equal(
          tf.ones_like(real_score), tf.round(tf.nn.sigmoid(real_score)))
      # correct = tf.equal(
      #    tf.zeros_like(fake_score), tf.round(tf.nn.sigmoid(fake_score)))
      disc_acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    else:
      correct = tf.equal(labels_disc, tf.round(tf.nn.sigmoid(all_score)))
      disc_acc = tf.reduce_mean(tf.cast(correct, tf.float32))

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
    tf.summary.scalar('disc_acc', disc_acc)
    tf.summary.scalar('gene_loss', gene_loss)
    tf.summary.scalar('disc_loss', disc_loss)

  else:
    tf.summary.histogram('real_score', real_score)
    tf.summary.histogram('fake_score', fake_score)

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
    print(' [*] Load SUCCESS')
  else:
    print(' [!] Load failed...')

  start_time = time.time()

  if FLAGS.is_test:
    index = 0
    overall_acc = []
    file_object = open(TEST_PATH, 'rb')
    for data_batch in read_by_batch(file_object, FLAGS.batch_size, [DATA_FRAME, DATA_DIM, 1]):
      if data_batch.shape[0] != FLAGS.batch_size:
        break
      noise_batch = np.random.uniform(-1, 1, [FLAGS.batch_size,
                                              NOISE_DIM]).astype(np.float32)
      summary, disc_acc_value, gene_loss_value, disc_loss_value = sess.run(
          [merged, disc_acc, gene_loss, disc_loss],
          feed_dict={
              real_data_holder: data_batch,
              input_noise_holder: noise_batch
          })
      writer.add_summary(summary, index)
      overall_acc = np.append(overall_acc, disc_acc_value)
      print('[Test %2d] G_loss: %.8f, D_loss: %.8f, D_accuracy: %.8f' %
            (index, gene_loss_value, disc_loss_value, disc_acc_value))
      index += 1
      if index >= FLAGS.epoch:
        break
    print('Overall accuracy is: ' + str(np.mean(overall_acc)))
    return

  for epoch in xrange(FLAGS.start_epoch, FLAGS.start_epoch + FLAGS.epoch):
    index = 0
    file_object = open(DATA_PATH, 'rb')
    print('Current Epoch is: ' + str(epoch))

    for data_batch in read_by_batch(file_object, FLAGS.batch_size, [DATA_FRAME, DATA_DIM, 1]):
      if data_batch.shape[0] != FLAGS.batch_size:
        break
      noise_batch = np.random.uniform(-1, 1, [FLAGS.batch_size,
                                              NOISE_DIM]).astype(np.float32)

      if epoch % FLAGS.sample == 0 and index == 0:
        # print data_batch[:10]
        summary, gene_loss_value, disc_loss_value = sess.run(
            [merged, gene_loss, disc_loss],
            feed_dict={
                real_data_holder: data_batch,
                input_noise_holder: noise_batch
            })
        writer.add_summary(summary, epoch)

        print('[Getting Sample...] G_loss: %2.8f, D_loss: %2.8f' %
              (gene_loss_value, disc_loss_value))

      if epoch % FLAGS.ckpt == 0 and index == 0:
        save(sess, saver, CHECKPOINT_DIR, counter)

      for _ in xrange(FLAGS.disc_iter):
        _, gene_loss_value, disc_loss_value = sess.run(
            [disc_train_op, gene_loss, disc_loss],
            feed_dict={
                real_data_holder: data_batch,
                input_noise_holder: noise_batch
            })

      for _ in xrange(FLAGS.gene_iter):
        _, gene_loss_value, disc_loss_value = sess.run(
            [gene_train_op, gene_loss, disc_loss],
            feed_dict={
                real_data_holder: data_batch,
                input_noise_holder: noise_batch
            })
      if index % 10 == 0:
        print(
            'Epoch: %3d batch: %4d time: %4.2f, G_loss: %2.8f, D_loss: %2.8f' %
            (epoch, index, time.time() - start_time, gene_loss_value,
             disc_loss_value))
      index += 1
      counter += 1


def print_args():
  print('dataname is:      ' + str(FLAGS.dataname))
  print('fresh_start is:   ' + str(FLAGS.fresh_start))
  print('learning_rate is: ' + str(FLAGS.learning_rate))
  print('epoch is:         ' + str(FLAGS.epoch))
  print('start_epoch is:   ' + str(FLAGS.start_epoch))
  print('is_test is:       ' + str(FLAGS.is_test))
  print('iwgan is:         ' + str(FLAGS.iwgan))
  print('old_param is:     ' + str(FLAGS.old_param))
  print('diff_lr is:       ' + str(FLAGS.diff_lr))
  print('old_lr is:        ' + str(FLAGS.old_lr))
  print('batch_size is:    ' + str(FLAGS.batch_size))
  print('ckpt is:          ' + str(FLAGS.ckpt))
  print('sample is:        ' + str(FLAGS.sample))
  print('gene_iter is:     ' + str(FLAGS.gene_iter))
  print('disc_iter is:     ' + str(FLAGS.disc_iter))
  print('gpu is:           ' + str(FLAGS.gpu))
  print('')


def main(_):

  print_args()

  if os.path.exists(CHECKPOINT_DIR) and FLAGS.fresh_start:
    shutil.rmtree(CHECKPOINT_DIR)
  elif not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
  if os.path.exists(LOG_DIR) and FLAGS.fresh_start:
    shutil.rmtree(LOG_DIR)
  elif not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
  if not os.path.exists(SAMPLE_DIR) and FLAGS.is_test:
    os.makedirs(SAMPLE_DIR)

  run_config = tf.ConfigProto(allow_soft_placement=True)
  run_config.gpu_options.allow_growth = True
  with tf.device('/gpu:' + str(FLAGS.gpu)):
    with tf.Session(config=run_config) as sess:
      run_model(sess)


if __name__ == '__main__':
  tf.app.run()
