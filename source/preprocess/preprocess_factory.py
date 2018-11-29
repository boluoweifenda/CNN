from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from preprocess import mnist_preprocess
from preprocess import cifar_preprocess

preprocess_map = {
  'mnist': mnist_preprocess.preprocess_image,
  'cifar': cifar_preprocess.preprocess_image,
}


def get_preprocess_fn(name):
  if name not in preprocess_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)
  func = preprocess_map[name]
  return func
