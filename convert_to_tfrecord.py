import os
import tensorflow as tf
import numpy as np
import scipy.io as sio


FILENAME = 'nina.tfrecords'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_data():
    writer = tf.python_io.TFRecordWriter(FILENAME)
    num = 0
    for i in xrange(27):
        # for j in xrange(1, 53):
        for k in xrange(10):
            matpath = 'data/%03d/001/%03d_001_%03d.mat' % (i, i, k)
            # print matpath
            source = sio.loadmat(matpath)
            data = np.array(source['data'])
            label = np.array(source['label'])
            repetition = np.array(source['repetition'])
            subject = np.array(source['subject'])
            for num, data_iter in enumerate(data):
                print num, data_iter.shape
                data_raw = data_iter.tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'data_raw': _bytes_feature(data_raw),
                            'label': _int64_feature(label),
                            'repetition': _int64_feature(repetition),
                            'subject': _int64_feature(subject)
                        }
                    )
                )
                writer.write(example.SerializeToString())

    writer.close()


def read_data(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'repetition': tf.FixedLenFeature([], tf.int64),
            'subject': tf.FixedLenFeature([], tf.int64)
        }
    )
    data = tf.decode_raw(features['data_raw'], tf.float64)
    data.set_shape([10])
    label = features['label']

    return data, label


def input(batch_size, num_epochs):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [FILENAME], num_epochs=num_epochs
        )
        data, label = read_data(filename_queue)
        datas, sparse_labels = tf.train.shuffle_batch(
            [data, label], batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    return datas, sparse_labels


def main(_):
    write_data()


if __name__ == '__main__':
    tf.app.run(main=main)
