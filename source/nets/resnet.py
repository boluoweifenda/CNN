import tensorflow as tf
from nets.net import Net


class ResNet(Net):

  def activation(self, x):
    if self.dataset in ['imagenet']:
      if self.width < 0.6:
        return tf.nn.relu6(x)  # 0.5x channels for mobile model
    return tf.nn.relu(x)

  # pre-activation residual block
  def residual(self, x, c_out, stride=1, bottleneck=False):
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

    if bottleneck:
      with tf.variable_scope('C0'):
        x = self.conv(x, 1, c_out / 4)
        x = self.batch_norm(x)
        x = self.activation(x)
      with tf.variable_scope('C1'):
        x = self.conv(x, 3, c_out / 4, stride)  # stride 2 in 3x3 conv following fb.resnet.torch
        x = self.batch_norm(x)
        x = self.activation(x)
      with tf.variable_scope('C2'):
        x = self.conv(x, 1, c_out)
    else:
      with tf.variable_scope('C0'):
        x = self.conv(x, 3, c_out, stride)
        x = self.batch_norm(x)
        x = self.activation(x)
      with tf.variable_scope('C1'):
        x = self.conv(x, 3, c_out)

    return x + shortcut

  def model(self, x):

    print('ResNet for %s dataset' % self.dataset)

    if self.dataset in ['cifar10', 'cifar100']:

      Repeat = 6  # total layer: 6n+2 / 9n+2
      bottleneck = True
      Stride = [1, 2, 2]
      Out = [16, 32, 64]

      if bottleneck:
        Out = [4 * i for i in Out]

      with tf.variable_scope('init'):
        x = self.conv(x, 3, Out[0])

      for i in range(len(Stride)):
        for j in range(Repeat):
          with tf.variable_scope('U%d-%d' % (i, j)):
            x = self.residual(x, Out[i], stride=Stride[i] if j is 0 else 1, bottleneck=bottleneck)

      with tf.variable_scope('global_avg_pool'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], name='fc')

      return x

    elif self.dataset in ['imagenet', 'tiny_imagenet']:

      self.width = 1.0
      bottleneck = True
      Repeat = [3, 4, 6, 3]  # 50
      Stride = [1, 2, 2, 2]
      Out = [256, 512, 1024, 2048]  # 1.0x

      Out = [int(self.width * i) for i in Out]

      with tf.variable_scope('init'):
        if self.dataset == 'imagenet':
          x = self.conv(x, 7, 64, stride=2)
          x = self.pool(x, type='MAX', ksize=3, stride=2)
        else:
          x = self.conv(x, 3, 64)

      for i in range(len(Stride)):
        for j in range(Repeat[i]):
          with tf.variable_scope('U%d-%d' % (i, j)):
            x = self.residual(x, Out[i], stride=Stride[i] if j is 0 else 1, bottleneck=bottleneck)

      with tf.variable_scope('global_avg_pool'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], name='fc', bias=True)

      return x
