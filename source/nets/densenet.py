import tensorflow as tf
from nets.net import Net


class DenseNet(Net):

  def _denseLayer(self, x, bottleneck, growth, drop_prob=0.):
    x_orth = x

    x = self.batch_norm(x)
    x = self.activation(x)

    if bottleneck:
      with tf.variable_scope('Bottleneck'):
        x = self.conv(x, 1, 4 * growth)
        x = self.dropout(x, drop_prob)
        x = self.batch_norm(x)
        x = self.activation(x)

    with tf.variable_scope('Conv'):
      x = self.conv(x, 3, growth)
      x = self.dropout(x, drop_prob)

    x = tf.concat([x_orth, x], axis=1)  # NCHW
    return x

  def _transitionLayer(self, x, reduction, drop_prob):
    c_out = int(self.get_shape(x)[1] * reduction)
    x = self.batch_norm(x)
    x = self.activation(x)
    x = self.conv(x, 1, c_out)
    x = self.dropout(x, drop_prob)
    x = self.pool(x, 'AVG', 2, 2, padding='VALID')
    return x

  def model(self, x):

    print('DenseNet for %s dataset' % self.dataset)

    if self.dataset in ['cifar10', 'cifar100']:

      num_dense = 9
      bottleneck = True
      growthRate = 48
      drop_prob = 0.
      reduction = 0.5
      num_block = 3

      with tf.variable_scope('init'):
        x = self.conv(x, 3, 2 * growthRate)

      for i in range(num_block):
        for j in range(num_dense):
          with tf.variable_scope('B%d_L%d' % (i, j)):
            x = self._denseLayer(x, bottleneck, growthRate, drop_prob=drop_prob)
        if i is num_block - 1: break
        with tf.variable_scope('T%d' % i):
          x = self._transitionLayer(x, reduction=reduction, drop_prob=drop_prob)

      with tf.variable_scope('global_avg_pool'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], bias=True, name='fc')

      return x

    elif self.dataset in ['imagenet']:

      bottleneck = True
      growthRate = 32
      drop_prob = 0.0
      reduction = 0.5
      stages = [6, 12, 24, 16]  # densenet121
      # stages = [6, 12, 32, 32]  # densenet169
      # stages = [6, 12, 48, 32]  # densenet201
      # stages = [6, 12, 64, 48]  # densenet264

      with tf.variable_scope('init'):
        x = self.conv(x, 7, 2 * growthRate, stride=2)
        x = self.pool(x, 'MAX', 3, 2)

      for i in range(len(stages)):
        for j in range(stages[i]):
          with tf.variable_scope('B%d_L%d' % (i, j)):
            x = self._denseLayer(x, bottleneck, growthRate, drop_prob)
        if i is len(stages) - 1: break
        with tf.variable_scope('T%d' % i):
          x = self._transitionLayer(x, reduction=reduction, drop_prob=drop_prob)

      with tf.variable_scope('global_avg_pool'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], name='fc')

      return x
