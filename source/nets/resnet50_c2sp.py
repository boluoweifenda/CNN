import tensorflow as tf
from nets.net import Net
import warnings


class ResNet(Net):

  def activation(self, x):
    return tf.nn.relu6(x)

  def symmetric_padding(self, x, ksize=2):

    assert ksize in [2]

    c_in = self.get_shape(x)[1]

    zeros = [
      [0, 1, 0, 1],
      [1, 0, 1, 0],
      [0, 1, 1, 0],
      [1, 0, 0, 1],
    ]

    num_pattern = len(zeros)
    mod = c_in % num_pattern
    if mod is not 0:
      warnings.warn('channel number %d can not be divided by %d' % (c_in, num_pattern))

    num_split = mod * [1 + c_in // num_pattern] + (num_pattern - mod) * [c_in // num_pattern]

    x_slide = tf.split(x, num_split, axis=1)
    x_pad = []

    for i in range(num_pattern):
      pad = tf.constant([[0, 0], [0, 0], zeros[i][:2], zeros[i][2:]])
      x_pad.append(tf.pad(x_slide[i], paddings=pad))
    x = tf.concat(x_pad,axis=1)

    return x

  def conv_sp(self, x, ksize, c_out, stride=1, bias=False, name='conv2x2_sp'):
    x = self.symmetric_padding(x, ksize=2)
    x = self.conv(x, ksize, c_out=c_out, stride=stride, padding='VALID', bias=bias, name=name)
    return x

  def conv_all(self, x, c_out, stride=1, mode='c3'):
    if mode=='c2':
      return self.conv(x, ksize=2, c_out=c_out, stride=stride, padding='SAME')
    elif mode=='c3':
      return self.conv(x, ksize=3, c_out=c_out, stride=stride, padding='SAME')
    elif mode in ['c2sp', 'c2sp_optim']:
      return self.conv_sp(x, ksize=2, c_out=c_out, stride=stride)
    else:
      raise NotImplementedError('Wrong mode named: %s' % mode)

  def residual(self, x, c_out, stride=1, bottleneck=1, mode='c3'):

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
      x = self.conv(x, 1, c_out / bottleneck)
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope('C1'):
      if stride is not 1:
        x = self.pool(x, 'AVG', ksize=2, stride=2)
      x = self.conv_all(x, c_out / bottleneck, stride=1, mode=mode)  # following fb.resnet.torch
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope('C2'):
      x = self.conv(x, 1, c_out)

    return x + shortcut

  def model(self, x):

    assert self.dataset is 'imagenet'
    print('resnet for %s dataset' % self.dataset)

    mode = 'c3'  # c3, c2, c2sp, c2sp_optim

    if mode is 'c2sp_optim':

      Repeat = [1, 2, 4, 8, 4]
      Stride = [1, 2, 2, 2, 2]
      Out = [48, 96, 192, 384, 768]
      bottleneck = [4, 4, 4, 4, 4]

      with tf.variable_scope('init'):
        x = self.conv(x, 3, 48, stride=2)

      for i in range(len(Stride)):
        for j in range(Repeat[i]):
          with tf.variable_scope('U%d-%d' % (i, j)):
            x = self.residual(x, Out[i], stride=Stride[i] if j is 0 else 1, mode=mode, bottleneck=bottleneck[i])

      with tf.variable_scope('last'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv(x, 1, 2 * Out[-1])

      with tf.variable_scope('global_avg_pool'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.dropout(x, 0.2)
        x = self.fc(x, self.shape_y[1], name='fc', bias=True)

      return x

    else:

      Repeat = [3, 4, 6, 3]
      Stride = [1, 2, 2, 2]
      Out = [128, 256, 512, 1024]  # 0.5x
      bottleneck = [4, 4, 4, 4]

      with tf.variable_scope('init'):
        x = self.conv(x, 7, 64, stride=2)
        x = self.pool(x, type='MAX', ksize=3, stride=2)

      for i in range(len(Stride)):
        for j in range(Repeat[i]):
          with tf.variable_scope('U%d-%d' % (i, j)):
            x = self.residual(x, Out[i], stride=Stride[i] if j is 0 else 1, mode=mode, bottleneck=bottleneck[i])

      with tf.variable_scope('global_avg_pool'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], name='fc', bias=True)

      return x


