from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from preprocess.preprocess_factory import get_preprocess_fn
import tensorflow as tf
from tensorflow.contrib import data


def get_batch(datasets, preprocess_name, is_training, batch_size, num_gpu=1, seed=None):
  with tf.device('/cpu:0'):

    num_class = datasets.num_class
    file_name = datasets.source
    feature = datasets.feature
    decoder = datasets.decoder
    name = datasets.description['name']

    image_preprocessing_fn = get_preprocess_fn(preprocess_name)

    dataset = tf.data.Dataset.from_tensor_slices(file_name)

    if is_training:
      # Shuffle the input files
      dataset = dataset.shuffle(len(file_name), seed=seed, reshuffle_each_iteration=True)

    '''  
    Convert to individual records.
    cycle_length = 8 means 8 files will be read and deserialized in parallel.
    This number is low enough to not cause too much contention on small systems
    but high enough to provide the benefits of parallelization. You may want
    to increase this number if you have a large number of CPU cores.
    '''

    cycle_length = min(10, len(file_name))
    dataset = dataset.apply(
      data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=cycle_length))

    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
      dataset = dataset.apply(data.shuffle_and_repeat(buffer_size=10000, seed=seed))
    else:
      dataset = dataset.repeat()

    def map_func(record):

      # Some images in imagenet are grayscaled.
      num_channel = 3 if name in ['imagenet', 'tiny_imagenet'] else 0

      if preprocess_name != 'inception_v2':
        parsed = tf.parse_single_example(record, feature)
        image = decoder(parsed['image/encoded'], num_channel)
        # Perform additional preprocessing on the parsed data.
        image = image_preprocessing_fn(image, datasets, is_training=is_training)
        label = parsed['image/class/label']
      else:
        from preprocess.inception_preprocess_v2 import parse_record
        image, label = parse_record(record, is_training)
      label = tf.one_hot(label, num_class)
      return image, label

    '''
    Parse the raw records into images and labels. Testing has shown that setting
    num_parallel_batches > 1 produces no improvement in throughput, since
    batch_size is almost always much greater than the number of CPU cores.    
    '''
    dataset = dataset.apply(
      data.map_and_batch(
        map_func=map_func,
        batch_size=batch_size,
        num_parallel_batches=1))

    '''
    Operations between the final prefetch and the get_next call to the iterator
    will happen synchronously during run time. We prefetch here again to
    background all of the above processing work and keep it out of the
    critical training path.    
    '''

    dataset = dataset.prefetch(buffer_size=32)
    iterator = dataset.make_one_shot_iterator()
    return iterator
