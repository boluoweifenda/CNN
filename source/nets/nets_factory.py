from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import lenet
from nets import mlp


nets_map = {
  'lenet': lenet.LeNet,
  'mlp': mlp.MLP,
}


def get_net_fn(name):
  if name not in nets_map:
    raise ValueError('Name of network unknown %s' % name)
  func = nets_map[name]
  return func

