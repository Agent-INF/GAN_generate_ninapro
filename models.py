import tensorflow as tf
from ops import fully_connect
from ops import conv2d
from ops import deconv2d
from ops import lrelu
from ops import batch_norm


def discriminator(data, batch_size, reuse=False):
  with tf.variable_scope('discriminator') as scope:
    if reuse:
      scope.reuse_variables()

    layer_num = 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(data, 16, [3, 3], [2, 2], name='conv_old')
      hidden = lrelu(batch_norm(hidden, name='bn_old'))
      #hidden = prelu(batch_norm(hidden, name='bn_old'))

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = conv2d(hidden, 32, [3, 3], [2, 2], name='conv_old')
      hidden = lrelu(batch_norm(hidden, name='bn_old'))
      #hidden = prelu(batch_norm(hidden, name='bn_old'))

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = fully_connect(tf.reshape(
          hidden, [batch_size, -1]), 1, name='fc_new')

    return hidden[:, 0]


def generator(noise, batch_size, hidden_shape, crop, is_train=True):
  with tf.variable_scope('generator') as scope:
    if not is_train:
      scope.reuse_variables()

    layer_num = 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = fully_connect(
          noise, hidden_shape[0][0] * hidden_shape[1][0] * 32, name='fc_new')
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train, name='bn_new'))
      hidden = tf.reshape(hidden, [-1, hidden_shape[0][0], hidden_shape[1][0], 32])

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = deconv2d(
          hidden, [batch_size, hidden_shape[0][1], hidden_shape[1][1], 16],
          [3, 3], [2, 2], name='conv_old')
      hidden = tf.nn.relu(batch_norm(hidden, train=is_train, name='bn_old'))

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = deconv2d(
          hidden, [batch_size, hidden_shape[0][2], hidden_shape[1][2], 1],
          [3, 3], [2, 2], name='conv_old')
      #hidden = tf.nn.sigmoid(hidden)

    layer_num += 1
    with tf.variable_scope('hidden' + str(layer_num)):
      hidden = tf.maximum(0.0024, hidden)

    return hidden[:, crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
