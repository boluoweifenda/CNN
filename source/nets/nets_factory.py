from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import mlp
from nets import lenet
from nets import vgg7
from nets import alexnet
from nets import resnet
from nets import densenet
from nets import mobilenet_v1
from nets import mobilenet_v2
from nets import shufflenet_v2
from nets import resnet50_c2sp
# from nets import shiftnet  # need to compile kernels in utils/active_shift

nets_map = {
  'mlp': mlp.MLP,
  'lenet': lenet.LeNet,
  'vgg7': vgg7.VGG7,
  'alexnet': alexnet.AlexNet,
  'resnet': resnet.ResNet,
  'densenet': densenet.DenseNet,
  'mobilenet_v1': mobilenet_v1.MobileNet,
  'mobilenet_v2': mobilenet_v2.MobileNet,
  'shufflenet_v2': shufflenet_v2.ShuffleNet,
  'resnet50_c2sp': resnet50_c2sp.ResNet,
  'shufflenet_test': shufflenet_test.ShuffleNet,
  # 'shiftnet': shiftnet.ShiftNet,
}


def get_net_fn(name):
  if name not in nets_map:
    raise ValueError('Name of network unknown %s' % name)
  func = nets_map[name]
  return func
