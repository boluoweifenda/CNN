from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from preprocess import mnist_preprocess
from preprocess import cifar_preprocess
from preprocess import cifar_slim_preprocess
from preprocess import alexnet_preprocess
from preprocess import inception_preprocess
from preprocess import inception_preprocess_v2

preprocess_map = {
  'mnist': mnist_preprocess.preprocess_image,
  'svhn': mnist_preprocess.preprocess_image,
  'cifar': cifar_preprocess.preprocess_image,
  'cifar_slim': cifar_slim_preprocess.preprocess_image,
  'alexnet': alexnet_preprocess.preprocess_image,
  'inception': inception_preprocess.preprocess_image,
  'inception_v2': inception_preprocess_v2.preprocess_image,
}


def get_preprocess_fn(name):
  if name not in preprocess_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)
  func = preprocess_map[name]
  return func
