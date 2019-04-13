import tensorflow as tf
from nets.net import Net


class DenseNet(Net):

  def _denseLayer(self, x, bottleneck, growthRate, drop_prob=0, first=False):
    x_orth = x

    if first is False:
      x = self._batch_norm(x)
      x = self._activation(x)

    if bottleneck:
      with tf.variable_scope('Bottleneck'):
        x = self._conv(x, 1, 4 * growthRate)
        x = self._dropout(x, drop_prob)
        x = self._batch_norm(x)
        x = self._activation(x)

    with tf.variable_scope('Conv'):
      x = self._conv(x, 3, growthRate)
      x = self._dropout(x, drop_prob)

    x = tf.concat([x_orth, x], axis=1)  # NCHW
    return x

  def _transitionLayer(self, x, reduction, drop_prob, last):
    c_out = int(self._shape(x)[1] * reduction)

    x = self._batch_norm(x)
    x = self._activation(x)
    if last:
      x = self._pool(x, 'GLO')
    else:
      x = self._conv(x, 1, c_out)
      x = self._dropout(x, drop_prob)
      x = self._pool(x, 'AVG', 2, 2, padding='VALID')
    return x

  def model(self, x):

    if self._shape(x)[-1] == 32:
      print('DenseNet for CIFAR dataset')

      depth = 100
      bottleneck = True
      num_block = 3
      N = (depth - 4) / num_block
      if bottleneck:
        N = int(N / 2)
      growthRate = 12
      drop_prob = 0.
      reduction = 0.5

      x = self.H[-1]

      with tf.variable_scope('init'):
        x = self._conv(x, 3, 2 * growthRate)

      for i in range(num_block):
        for j in range(N):
          with tf.variable_scope('B%d_L%d' % (i, j)):
            x = self._denseLayer(x, bottleneck, growthRate, drop_prob)
        with tf.variable_scope('T%d' % i):
          x = self._transitionLayer(x, reduction=reduction, drop_prob=drop_prob, last=True if i == num_block - 1 else False)
      with tf.variable_scope('logit'):
        x = self._fc(x, self.shape_y[1], bias=True, name='fc')

      return x

    elif self._shape(x)[-1] == 224:
      print('DenseNet for ImageNet dataset')

      bottleneck = True
      growthRate = 32
      drop_prob = 0.0
      reduction = 0.5
      stages = [6, 12, 24, 16]  # densenet121
      # stages = [6, 12, 32, 32]  # densenet169
      # stages = [6, 12, 48, 32]  # densenet201
      # stages = [6, 12, 64, 48]  # densenet264

      x = self.H[-1]
      with tf.variable_scope('init'):
        x = self._conv(x, 7, 2 * growthRate, stride=2)
        x = self._batch_norm(x)
        x = self._activation(x)
        x = self._pool(x, 'MAX', 3, 2)

      for i in range(len(stages)):
        for j in range(stages[i]):
          with tf.variable_scope('B%d_L%d' % (i, j)):
            x = self._denseLayer(x, bottleneck, growthRate, drop_prob, first=True if (i + j) == 0 else False)
        with tf.variable_scope('T%d' % i):
          x = self._transitionLayer(x, reduction=reduction, drop_prob=drop_prob, last=True if i == len(stages) - 1 else False)

      with tf.variable_scope('logit'):
        x = self._fc(x, self.shape_y[1], name='fc')

      return x

    else:
      assert False, 'Unknown image size'



