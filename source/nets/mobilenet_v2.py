import tensorflow as tf
from nets.net import Net


class MobileNet(Net):

  def activation(self, x):
    return tf.nn.relu6(x)

  def _expanded_conv(self, x, ksize, c_out, stride=1, expansion_size=6, padding='SAME',
                    name='expanded_conv'):
    x_orig = x
    c_in = self.get_shape(x)[1]
    if expansion_size != 1:
      with tf.variable_scope(name + '_expansion'):
        x = self.conv(x, 1, c_in * expansion_size, stride=1, padding=padding)
        x = self.batch_norm(x)
        x = self.activation(x)
    with tf.variable_scope(name + '_depthwise'):
      x = self.depthwise_conv(x, ksize, channel_multiplier=1, stride=stride, padding=padding)
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope(name + '_projection'):
      x = self.conv(x, 1, c_out, stride=1, padding=padding)
      x = self.batch_norm(x)
    if stride == 1 and c_in == c_out:
      x = x + x_orig
    return x

  def model(self, x):

    if self.dataset in ['cifar10', 'cifar100']:
      num_residual = 12
      Stride = [1, 2, 2]
      Out = [16, 32, 64]
      Expansion = [6, 6, 6]

      with tf.variable_scope('init'):
        x = self.conv(x, 3, Out[0])
        x = self.batch_norm(x)
        x = self.activation(x)

      for i in range(len(Stride)):
        for j in range(num_residual):
          x = self._expanded_conv(x, 3, Out[i], stride=Stride[i] if j == 0 else 1, expansion_size=Expansion[i], name='block%d-%d' % (i,j))

      with tf.variable_scope('last'):
        x = self.conv(x, 1, Out[-1]*6)
        x = self.batch_norm(x)
        x = self.activation(x)
      with tf.variable_scope('global_avg_pool'):
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], name='fc')

      return x

    elif self.dataset in ['imagenet', 'tiny_imagenet']:

      Stride = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
      # Out = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64,  96,  96,  96, 160, 160, 160, 320, 1280]  # 1.0x
      Out = [48, 24, 32, 32, 48, 48, 48, 88, 88, 88, 88, 136, 136, 136, 224, 224, 224, 448, 1792]  # 1.4x
      Expansion = [1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

      with tf.variable_scope('init'):
        if self.dataset == 'imagenet':
          x = self.conv(x, 3, Out[0], stride=2)
        else:
          Stride, Out, Expansion = Stride[2:], Out[2:], Expansion[2:]
          x = self.conv(x, 3, Out[0])
        x = self.batch_norm(x)
        x = self.activation(x)

      for i in range(len(Stride)):
        x = self._expanded_conv(x, 3, Out[i+1], Stride[i], Expansion[i], name='block%d' % i)

      with tf.variable_scope('last'):
        x = self.conv(x, 1, Out[-1])
        x = self.batch_norm(x)
        x = self.activation(x)
      with tf.variable_scope('global_avg_pool'):
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.dropout(x, 0.2)
        x = self.fc(x, self.shape_y[1], name='fc', bias=True)

      return x



