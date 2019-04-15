from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import lenet
from nets import vgg7
from nets import alexnet
from nets import mobilenet_v1
from nets import mobilenet_v2
from nets import resnet
from nets import densenet
from nets import mlp
from nets import shufflenet_v2


nets_map = {
  'lenet': lenet.LeNet,
  'vgg7': vgg7.VGG7,
  'alexnet': alexnet.AlexNet,
  'mobilenet_v1': mobilenet_v1.MobileNet,
  'mobilenet_v2': mobilenet_v2.MobileNet,
  'resnet': resnet.ResNet,
  'densenet': densenet.DenseNet,
  'mlp': mlp.MLP,
  'shufflenet_v2': shufflenet_v2.ShuffleNet,
}

def get_net_fn(name):
  if name not in nets_map:
    raise ValueError('Name of network unknown %s' % name)
  func = nets_map[name]
  return func
