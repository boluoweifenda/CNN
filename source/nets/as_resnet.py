import tensorflow as tf
import numpy as np
from nets.net import Net
from utils.active_shift.active_shift2d_op import active_shift2d


class ResNet(Net):

  def activation(self, x):
    return tf.nn.relu6(x)

  def _group_shift(self, x, active=False):

    c_in = self.get_shape(x)[1]

    if self.shift_init in ['heur9', 'heur8', 'heur124']:
      kernel = [
        [-1, -1],
        [ 0, -1],
        [ 1, -1],
        [-1,  0],
        [ 0,  0],
        [ 1,  0],
        [-1,  1],
        [ 0,  1],
        [ 1,  1],
      ]

      if self.shift_init is 'heur8':
        kernel.pop(4)
      if self.shift_init is 'heur124':
        ratio = [1,2,1,2,4,2,1,2,1]
      else:
        ratio = [1]*len(kernel)

      num_ratio = sum(ratio)
      assert c_in % num_ratio == 0, 'channel can not be divided by %d' % num_ratio

      kernels = []
      for i in range(len(kernel)):
        repeat = int(c_in/num_ratio*ratio[i])
        kernels += repeat*[kernel[i]]

      kernels = np.transpose(np.array(kernels, dtype=np.float32))

    elif self.shift_init is 'uniform':
      kernels = np.random.uniform(low=-1, high=1, size=[2, c_in])

    if active:
      kernels = tf.get_variable(name='shift', shape=[2, c_in], initializer=tf.initializers.constant(kernels), dtype=tf.float32)
    else:
      kernels = tf.constant(kernels, dtype=tf.float32)
    x = active_shift2d(x, kernels, strides=[1, 1, 1, 1], paddings=[0, 0, 0, 0])

    return x

  # pre-activation residual block
  def _residual(self, x, c_out, stride=1, active=True, mode='shift'):
    c_in = self.get_shape(x)[1]
    shortcut = x

    x = self.batch_norm(x)
    x = self.activation(x)

    if stride is not 1 or c_in != c_out:
      # The conv1x1 projection shortcut should come after the first batchnorm and ReLU
      shortcut = x
      if stride is not 1:
        # suggested by "Bag of Tricks for Image Classification with Convolutional Neural Networks"
        shortcut = self.pool(shortcut, 'AVG', ksize=2, stride=2)
      if c_in != c_out:
        shortcut = self.conv(shortcut, 1, c_out)

    with tf.variable_scope('C0'):
      x = self.conv(x, 1, c_out)
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope('C1'):
      # x = self.conv(x, 3, c_out / 4, stride)  # stride 2 in 3x3 conv following fb.resnet.torch
      if stride is not 1:
        x = self.pool(x, 'AVG', ksize=2, stride=2)
      x = self._group_shift(x, active=active)
    with tf.variable_scope('C2'):
      x = self.conv(x, 1, c_out)

    return x + shortcut

  def model(self, x):

    active = True
    self.shift_init = 'uniform'
    mode = 'shift'

    num_residual = [1, 3, 4, 6, 3]  # 50
    strides = [1, 2, 2, 2, 2]
    filters = [68, 68, 136, 272, 544]  # 0.5x

    with tf.variable_scope('init'):
      x = self.conv(x, 3, filters[0], stride=2)

    for i in range(len(num_residual)):
      for j in range(num_residual[i]):
        with tf.variable_scope('U%d-%d' % (i, j)):
          x = self._residual(x, filters[i], stride=strides[i] if j is 0 else 1, active=active)

    with tf.variable_scope('global_avg_pool'):
      x = self.batch_norm(x)
      x = self.activation(x)
      x = self.pool(x, 'GLO')

    with tf.variable_scope('logit'):
      x = self.fc(x, self.shape_y[1], name='fc', bias=True)

    return x












