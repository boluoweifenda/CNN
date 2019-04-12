from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import tensorflow as tf

from datasets import dataset

FILE_PATTERN = 'fashion_%s.tfrecord'
FILE_LABEL = 'labels.txt'
NUM_SAMPLE = {'train': 60000, 'test': 10000}
NUM_CLASSE = 10

DESCRIPTION = {
  'name': 'MNIST-FASHION',
  'image': 'A [28 x 28 x 1] grayscale image.',
  'label': 'A single integer between 0 and 9',
  'num_file_train': 1,
  'num_file_test': 1,
}


def get_split(split_name, dataset_dir, file_pattern=None):

  if split_name not in NUM_SAMPLE:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = FILE_PATTERN
  file_pattern = dataset_dir + file_pattern % split_name

  source = glob.glob(file_pattern)
  decoder = tf.image.decode_png
  label = dataset.read_label_file(dataset_dir + FILE_LABEL)
  feature = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/class/label': tf.FixedLenFeature(
      [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  return dataset.Dataset(
    source=source,
    feature=feature,
    decoder=decoder,
    num_sample=NUM_SAMPLE[split_name],
    num_class=NUM_CLASSE,
    label=label,
    description=DESCRIPTION)
