from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def preprocess_image(image, dataset, is_training=False):
  image.set_shape([28, 28, 1])
  image = tf.image.convert_image_dtype(image, tf.float32)
  return image

