import math
import numpy as np
import tensorflow as tf


def concat(tensors, axis, *args, **kwargs):
  return tf.concat(tensors, axis, *args, **kwargs)


def batch_norm(_x, train=True, epsilon=1e-5, momentum=0.9, name="batch_norm"):
  return tf.contrib.layers.batch_norm(
      _x,
      decay=momentum,
      updates_collections=None,
      epsilon=epsilon,
      scale=True,
      is_training=train,
      scope=name)


def conv_cond_concat(_x, _y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = _x.get_shape()
  y_shapes = _y.get_shape()
  return concat(
      [_x,
       _y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_,
           output_dim,
           k_h=4,
           k_w=4,
           d_h=2,
           d_w=2,
           stddev=0.02,
           padding="SAME",
           name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable(
        'weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable(
        'biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def deconv2d(input_,
             output_shape,
             k_h=4,
             k_w=4,
             d_h=2,
             d_w=2,
             stddev=0.02,
             padding="SAME",
             name="deconv2d",
             with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable(
        'weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
        initializer=tf.random_normal_initializer(stddev=stddev))

    deconv = tf.nn.conv2d_transpose(
        input_,
        w,
        output_shape=output_shape,
        strides=[1, d_h, d_w, 1],
        padding=padding)

    biases = tf.get_variable(
        'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv


def lrelu(_x, leak=0.2):
  return tf.maximum(_x, leak * _x)


def prelu(_x, name='prelu'):
  with tf.variable_scope(name):
    alphas = tf.get_variable(
        'alpha',
        _x.get_shape()[-1],
        initializer=tf.constant_initializer(0.0),
        dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = tf.multiply(alphas, (_x - tf.abs(_x))) * 0.5
  return pos + neg


def pooling(_x,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pooling',
            pooltype='max'):
  if pooltype == 'max':
    return tf.nn.max_pool(
        _x, ksize=ksize, strides=strides, padding=padding, name=name)
  elif pooltype == 'avg':
    return tf.nn.avg_pool(
        _x, ksize=ksize, strides=strides, padding=padding, name=name)
  else:
    print 'pooling type error!'
    return


def linear(input_,
           output_size,
           scope=None,
           stddev=0.02,
           bias_start=0.0,
           with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable(
        "weights", [shape[1], output_size],
        tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable(
        "biases", [output_size],
        initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias


def channel_wise_fc(input_, stddev=0.02, bias_start=0.0,
                    name='channel_wise_fc'):
  _, width, height, channel = input_.get_shape().as_list()
  input_reshape = tf.reshape(input_, [-1, width * height, channel])
  input_transpose = tf.transpose(input_reshape, [2, 0, 1])

  with tf.variable_scope(name):
    W = tf.get_variable(
        "weights",
        shape=[channel, width * height, width * height],
        initializer=tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable(
        "biases", [channel, 1, width * height],
        initializer=tf.constant_initializer(bias_start))
    output = tf.matmul(input_transpose, W) + bias

  output_transpose = tf.transpose(output, [1, 2, 0])
  output_reshape = tf.reshape(output_transpose, [-1, height, width, channel])

  return output_reshape
