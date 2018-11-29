from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import mnist
from datasets import fashion


datasets_map = {
  'mnist': mnist,
  'fashion': fashion,
}

dir_map = {
  'mnist': '../data/mnist/',
  'fashion': '../data/fashion/',
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
