from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import mnist
from datasets import fashion
from datasets import cifar10
from datasets import cifar100
from datasets import imagenet


datasets_map = {
  'mnist': mnist,
  'fashion': fashion,
  'cifar10': cifar10,
  'cifar100': cifar100,
  'imagenet': imagenet,
}

dir_map = {
  'mnist': '../../../datasets/mnist/',
  'fashion': '../../../datasets/fashion/',
  'cifar10': '../../../datasets/cifar10/',
  'cifar100': '../../../datasets/cifar100/',
  'imagenet': '/imagenet/TFRecord/',  # we move imagenet dataset to SSD for better reading speed
  'imagenet256': '/imagenet/TFRecord256/',  # we move imagenet dataset to SSD for better reading speed
}


def get_dataset(name, split, file_pattern=None):
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  if name not in dir_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(
      split,
      dir_map[name],
      file_pattern)
