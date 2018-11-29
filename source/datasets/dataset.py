from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Dataset(object):
  """Represents a Dataset specification."""

  def __init__(self, source, feature, decoder, num_sample, num_class, label, description, **kwargs):

    kwargs['source'] = source
    kwargs['feature'] = feature
    kwargs['decoder'] = decoder
    kwargs['num_sample'] = num_sample
    kwargs['num_class'] = num_class
    kwargs['label'] = label
    kwargs['description'] = description
    self.__dict__.update(kwargs)


def read_label_file(path):
  lines = open(path).readlines()
  dict_label = {}
  for line in lines:
    split = line.find(':')
    index = line[:split]
    name = line[split+1:-1]  # last two are '/n'
    dict_label[int(index)] = name
  return dict_label