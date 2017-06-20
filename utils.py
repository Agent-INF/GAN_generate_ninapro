import os
import numpy as np
import tensorflow as tf


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
    counter = int(next(re.finditer("([0-9]+)(?!.*[0-9])", ckpt_name)).group(0))
    print " [*] Success to read {}".format(ckpt_name)
    return True, counter
  else:
    print ' [*] Failed to find a checkpoint'
    return False, 0


def read_by_batch(file_object, batch_size, data_shape):
  """read file one batch at a time, data shape shoud be NHWC"""
  assert len(data_shape) == 3, 'Wrong data_shape: ' + str(data_shape)
  while True:
    size = data_shape[0] * data_shape[1]
    batch = np.fromfile(file_object, dtype=np.float64, count=size * batch_size)
    if batch is None:
      break
    data = np.reshape(batch, (-1, data_shape[0], data_shape[1], data_shape[2]))
    yield data
