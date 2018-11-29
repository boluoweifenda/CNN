from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def preprocess_image(image, dataset, is_training=False):
  image = tf.to_float(image)
  image = image/255.
  image.set_shape([28, 28, 1])
  return image

