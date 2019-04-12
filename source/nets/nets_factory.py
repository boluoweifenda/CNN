from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import lenet
from nets import vgg7
from nets import alexnet
from nets import mobilenet_v1
from nets import mobilenet_v2
from nets import resnet
from nets import resnet_test
from nets import densenet
from nets import densenet_test
from nets import nasnet_test
from nets import mlp
from nets import shiftnet
from nets import shufflenet_v2
from nets import resnet_conv2x2_fig3


nets_map = {
  'lenet': lenet.LeNet,
  'vgg7': vgg7.VGG7,
  'alexnet': alexnet.AlexNet,
  'mobilenet_v1': mobilenet_v1.MobileNet,
  'mobilenet_v2': mobilenet_v2.MobileNet,
  'resnet': resnet.ResNet,
  'resnet_test': resnet_test.ResNet,
  'densenet': densenet.DenseNet,
  'densenet_test': densenet_test.DenseNet,
  'nasnet': nasnet_test.NASNet,
  'mlp': mlp.MLP,
  'shiftnet': shiftnet.ShiftNet,
  'shufflenet_v2': shufflenet_v2.ShuffleNet,
  'resnet_conv2x2_fig3':resnet_conv2x2_fig3.ResNet

}

def get_net_fn(name):
  if name not in nets_map:
    raise ValueError('Name of network unknown %s' % name)
  func = nets_map[name]
  return func
