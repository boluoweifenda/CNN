from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from preprocess.preprocess_factory import get_preprocess_fn
import tensorflow as tf
from tensorflow.contrib import slim


def get_batch(dataset, preprocess_name, is_training, batch_size, num_gpu=1, seed=None):
  with tf.device('/cpu:0'):

    num_file_train = dataset.items_to_descriptions['num_file_train']
    num_classes = dataset.num_classes

    num_readers = min(num_file_train, 4) if is_training else 1
    provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      common_queue_capacity=20 * batch_size,
      common_queue_min=10 * batch_size)

    [image, label] = provider.get(['image', 'label'])
    image_preprocessing_fn = get_preprocess_fn(preprocess_name)
    image = image_preprocessing_fn(image, is_training=is_training)

    if is_training:
      images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=8 * num_gpu,
        capacity=32 * batch_size,
        min_after_dequeue=16 * batch_size,
        seed=seed)
    else:
      images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4 * num_gpu,
        capacity=8 * batch_size)

    labels = slim.one_hot_encoding(labels, num_classes)

    batch_queue = slim.prefetch_queue.prefetch_queue(
      [images, labels], capacity=4 * num_gpu)

    return batch_queue
