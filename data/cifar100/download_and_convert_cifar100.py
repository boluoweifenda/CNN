from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from six.moves import cPickle
from six.moves import urllib
import tensorflow as tf

# The URL where the CIFAR data can be downloaded.
_DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

# The number of training files.
_NUM_TRAIN_FILES = 1

# The height and width of each image.
_IMAGE_SIZE = 32

# The names of the classes.
_CLASS_NAMES = [
  'apple',
  'aquarium_fish',
  'baby',
  'bear',
  'beaver',
  'bed',
  'bee',
  'beetle',
  'bicycle',
  'bottle',
  'bowl',
  'boy',
  'bridge',
  'bus',
  'butterfly',
  'camel',
  'can',
  'castle',
  'caterpillar',
  'cattle',
  'chair',
  'chimpanzee',
  'clock',
  'cloud',
  'cockroach',
  'couch',
  'crab',
  'crocodile',
  'cup',
  'dinosaur',
  'dolphin',
  'elephant',
  'flatfish',
  'forest',
  'fox',
  'girl',
  'hamster',
  'house',
  'kangaroo',
  'keyboard',
  'lamp',
  'lawn_mower',
  'leopard',
  'lion',
  'lizard',
  'lobster',
  'man',
  'maple_tree',
  'motorcycle',
  'mountain',
  'mouse',
  'mushroom',
  'oak_tree',
  'orange',
  'orchid',
  'otter',
  'palm_tree',
  'pear',
  'pickup_truck',
  'pine_tree',
  'plain',
  'plate',
  'poppy',
  'porcupine',
  'possum',
  'rabbit',
  'raccoon',
  'ray',
  'road',
  'rocket',
  'rose',
  'sea',
  'seal',
  'shark',
  'shrew',
  'skunk',
  'skyscraper',
  'snail',
  'snake',
  'spider',
  'squirrel',
  'streetcar',
  'sunflower',
  'sweet_pepper',
  'table',
  'tank',
  'telephone',
  'television',
  'tiger',
  'tractor',
  'train',
  'trout',
  'tulip',
  'turtle',
  'wardrobe',
  'whale',
  'willow_tree',
  'wolf',
  'woman',
  'worm',
]


LABELS_FILENAME = 'labels.txt'


def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):

  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names


def _add_to_tfrecord(filename, tfrecord_writer, offset=0):

  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info < (3,):
      data = cPickle.load(f)
    else:
      data = cPickle.load(f, encoding='bytes')

  images = data[b'data']
  num_images = images.shape[0]

  images = images.reshape((num_images, 3, 32, 32))
  labels = data[b'fine_labels']

  with tf.Graph().as_default():
    image_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_image = tf.image.encode_png(image_placeholder)

    with tf.Session('') as sess:

      for j in range(num_images):
        sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
            filename, offset + j + 1, offset + num_images))
        sys.stdout.flush()

        image = np.squeeze(images[j]).transpose((1, 2, 0))
        label = labels[j]

        png_string = sess.run(encoded_image,
                              feed_dict={image_placeholder: image})

        example = image_to_tfexample(
            png_string, b'png', _IMAGE_SIZE, _IMAGE_SIZE, label)
        tfrecord_writer.write(example.SerializeToString())

  return offset + num_images


def _get_output_filename(dataset_dir, split_name):
  return '%s/cifar100_%s.tfrecord' % (dataset_dir, split_name)


def _download_and_uncompress_dataset(dataset_dir):

  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(_DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  else:
    print('File %s exists' % filename)

  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def run(dataset_dir=None):

  if dataset_dir is None:
    dataset_dir = os.getcwd()


  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train')
  testing_filename = _get_output_filename(dataset_dir, 'test')

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  _download_and_uncompress_dataset(dataset_dir)

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    offset = 0
    for i in range(_NUM_TRAIN_FILES):
      filename = os.path.join(dataset_dir,
                              'cifar-100-python',
                              'train')  # 1-indexed.
      offset = _add_to_tfrecord(filename, tfrecord_writer, offset)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    filename = os.path.join(dataset_dir,
                            'cifar-100-python',
                            'test')
    _add_to_tfrecord(filename, tfrecord_writer)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  write_label_file(labels_to_class_names, dataset_dir, LABELS_FILENAME)

  print('\nFinished converting the cifar100 dataset!')


if __name__ == '__main__':

  # use cpu to convert data
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  run()
