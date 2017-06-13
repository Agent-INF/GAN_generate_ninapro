"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('data/mnist', one_hot=True)
