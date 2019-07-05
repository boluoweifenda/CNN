import tensorflow as tf
from nets.net import Net


class ResNet(Net):

  # pre-activation residual block
  def _residual(self, x, c_out, stride=1, bottleneck=False):
    c_in = self.get_shape(x)[1]
    orig_x = x

    if bottleneck:
      with tf.variable_scope('S0'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv(x, 1, c_out / 4)
      with tf.variable_scope('S1'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv(x, 3, c_out / 4, stride)  # stride 2 in 3x3 conv following fb.resnet.torch
      with tf.variable_scope('S2'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv(x, 1, c_out)
    else:
      with tf.variable_scope('S0'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv(x, 3, c_out, stride)
      with tf.variable_scope('S1'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv(x, 3, c_out)

    # x = self.squeeze_and_excitation(x, r=16)

    with tf.variable_scope('SA'):
      if stride is not 1:
        # suggested by "Bag of Tricks for Image Classification with Convolutional Neural Networks"
        orig_x = self.pool(orig_x, 'AVG', ksize=2, stride=2)
      if c_in != c_out:
        orig_x = self.conv(orig_x, 1, c_out)
      x += orig_x

    return x

  def model(self, x):

    print('ResNet for %s dataset' % self.dataset)

    if self.dataset in ['cifar10', 'cifar100']:

      num_residual = 6  # totoal layer: 6n+2 / 9n+2
      bottleneck = True
      strides = [1, 2, 2]
      filters = [16, 32, 64]

      if bottleneck:
        filters = [4 * i for i in filters]

      with tf.variable_scope('init'):
        x = self.conv(x, 3, filters[0])

      for i in range(len(filters)):
        for j in range(num_residual):
          with tf.variable_scope('U%d-%d' % (i, j)):
            x = self._residual(x, filters[i], stride=strides[i] if j == 0 else 1, bottleneck=bottleneck)

      with tf.variable_scope('global_avg_pool'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], name='fc')

      return x

    elif self.dataset in ['imagenet', 'tiny_imagenet']:

      num_residual = [3, 4, 6, 3]  # 50
      # num_residual = [3, 4, 23, 3]  # 101
      strides = [1, 2, 2, 2]
      # filters = [128, 256, 512, 1024]  # 0.5x
      filters = [256, 512, 1024, 2048]  # 1.0x
      bottleneck = True

      with tf.variable_scope('init'):
        if self.dataset == 'imagenet':
          x = self.conv(x, 7, 64, stride=2)
          x = self.pool(x, type='MAX', ksize=3, stride=2)
        else:
          x = self.conv(x, 3, 64)

      for i in range(len(num_residual)):
        for j in range(num_residual[i]):
          with tf.variable_scope('U%d-%d' % (i, j)):
            x = self._residual(x, filters[i], stride=strides[i] if j is 0 else 1, bottleneck=bottleneck)

      with tf.variable_scope('global_avg_pool'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], name='fc', bias=True)

      return x
