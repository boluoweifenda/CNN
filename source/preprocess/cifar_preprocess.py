from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from preprocess.preprocess_ops import channel_normalize
from preprocess.preprocess_ops import cutout

CUTOUT = False
NORMALIZATION_FIRST = False


def preprocess_image(image, dataset, is_training=False):

  print('cifar preprocess, cutout=%s, normalization_first=%s' %(CUTOUT, NORMALIZATION_FIRST))

  name = dataset.description['name']
  image.set_shape([32, 32, 3])
  image = tf.cast(image, dtype=tf.float32)

  if name is 'cifar10':
    mean = [125.30694, 122.95031, 113.86539]
    std = [62.993233, 62.08874, 66.70485]
  elif name is 'cifar100':
    mean = [129.30428, 124.07023, 112.43411]
    std = [68.17024, 65.391785, 70.4184]
  else:
    raise NameError('Only for cifar10 or cifar100 dataset')

  if NORMALIZATION_FIRST:
    image = channel_normalize(image, mean, std)

  if is_training:
    image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    image = tf.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)

    if CUTOUT:
      image = cutout(image, 16, 16)

  if not NORMALIZATION_FIRST:
    image = channel_normalize(image, mean, std)

  return image
