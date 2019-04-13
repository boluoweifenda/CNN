import tensorflow as tf
from nets.net import Net


class ResNet(Net):

  def _residual(self, x, c_out, stride=1, bottleneck=False, first=False):
    c_in = self._shape(x)[1]
    orig_x = x

    # pre-activation residual block
    if first is False:
      with tf.variable_scope('S0'):
        x = self._batch_norm(x)
        x = self._activation(x)
    if bottleneck:
      with tf.variable_scope('S0'):
        x = self._conv(x, 1, c_out / 4)
      with tf.variable_scope('S1'):
        x = self._batch_norm(x)
        x = self._activation(x)
        # we use stride 2 in the 3x3 conv when using bottleneck following fb.resnet.torch
        x = self._conv(x, 3, c_out / 4, stride)
      with tf.variable_scope('S2'):
        x = self._batch_norm(x)
        x = self._activation(x)
        x = self._conv(x, 1, c_out)
    else:
      with tf.variable_scope('S0'):
        x = self._conv(x, 3, c_out, stride)
      with tf.variable_scope('S1'):
        x = self._batch_norm(x)
        x = self._activation(x)
        x = self._conv(x, 3, c_out)

    with tf.variable_scope('SA'):
      if stride is not 1 or c_in != c_out:
        # # Option B
        orig_x = self._conv(orig_x, 1, c_out, stride)
      x += orig_x

    return x

  def model(self, x):

    if self._shape(x)[-1] == 32:
      print('ResNet for cifar dataset')

      num_residual = 18  # totoal layer: 6n+2 / 9n+2
      strides = [1, 2, 2]
      filters = [16, 32, 64]
      bottleneck = False

      if bottleneck:
        filters = [4 * i for i in filters]

      with tf.variable_scope('init'):
        x = self._conv(x, 3, 16)

      for i in range(len(strides)):
        with tf.variable_scope('U%d-0' % i):
          x = self._residual(x, filters[i], strides[i], bottleneck)
        for j in range(1, num_residual):
          with tf.variable_scope('U%d-%d' % (i, j)):
            x = self._residual(x, filters[i], 1, bottleneck)

      with tf.variable_scope('global_avg_pool'):
        x = self._batch_norm(x)
        x = self._activation(x)
        x = self._pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self._fc(x, self.shape_y[1], name='fc')

      return x

    else:
      print('ResNet for ImageNet dataset')

      with tf.variable_scope('init'):
        x = self._conv(x, 7, 64, stride=2)
        x = self._batch_norm(x)
        x = self._activation(x)
        x = self._pool(x, type='MAX', ksize=3, stride=2)

      num_residual = [3, 4, 6, 3]
      # num_residual = [3, 4, 23, 3]
      # num_residual = [3, 8, 36, 3]
      strides = [1, 2, 2, 2]
      filters = [256, 512, 1024, 2048]
      bottleneck = True

      for i in range(len(strides)):
        with tf.variable_scope('U%d-0' % i):
          x = self._residual(x, filters[i], strides[i], bottleneck, first=True if i == 0 else False)
        for j in range(1, num_residual[i]):
          with tf.variable_scope('U%d-%d' % (i, j)):
            x = self._residual(x, filters[i], 1, bottleneck)

      with tf.variable_scope('global_avg_pool'):
        x = self._batch_norm(x)
        x = self._activation(x)
        x = self._pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self._fc(x, self.shape_y[1], name='fc')

      return x



