from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from preprocess.preprocess_ops import *


def preprocess_image(image, dataset, is_training=False):

  name = dataset.description['name']
  if name is 'imagenet':
    print('Please use dataset: imagenet256, which has been resize to 256x256 already to speed up training on alexnet')
    image = aspect_preserving_resize(image, 256)

  if is_training:
    image = tf.random_crop(image, [224, 224, 3])
    image = tf.image.random_flip_left_right(image)

  else:
    image = central_crop(image, 224, 224)

  return image
