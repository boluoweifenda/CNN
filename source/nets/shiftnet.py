import tensorflow as tf
from nets.net import Net


class ShiftNet(Net):

  def _group_shift(self, x):

    zeros = [
      [0, 2, 0, 2],
      [0, 2, 1, 1],
      [0, 2, 2, 0],
      [1, 1, 0, 2],
      [1, 1, 1, 1],
      [1, 1, 2, 0],
      [2, 0, 0, 2],
      [2, 0, 1, 1],
      [2, 0, 2, 0],
    ]

    num_pattern = len(zeros)
    c_in = self.get_shape(x)[1]
    assert c_in > num_pattern and c_in % num_pattern == 0, 'channel can not be divided by %d' % num_pattern

    x_slide = tf.split(x, num_pattern, axis=1)
    x_pad = []

    for i in range(num_pattern):
      pad = tf.constant([[0, 0], [0, 0], zeros[i][:2], zeros[i][2:]])
      x_pad.append(tf.pad(x_slide[i], paddings=pad))
    x = tf.concat(x_pad, axis=1)
    x = x[..., 1:-1, 1:-1]

    return x

  def _CSC(self, x, c_out, stride=1, expansion=6):
    c_in = self.get_shape(x)[1]
    orig_x = x

    with tf.variable_scope('C0'):
      x = self.batch_norm(x)
      x = self.activation(x)
      x = self.conv(x, 1, c_in * expansion, stride=1)

    x = self._group_shift(x)

    with tf.variable_scope('C1'):
      x = self.batch_norm(x)
      x = self.activation(x)
      x = self.conv(x, 1, c_in, stride=stride)

    with tf.variable_scope('SA'):
      if stride is not 1 or c_in != c_out:
        orig_x = self.pool(orig_x, type='AVG', ksize=3, stride=2)
        x = tf.concat([x, orig_x], axis=1)
      else:
        x += orig_x

    return x

  def model(self, x):

    print('ShiftNet for %s dataset' % self.dataset)

    if self.dataset in ['cifar10', 'cifar100']:

      num_residual = 9  # totoal layer: 6n+2
      strides = [1, 2, 2]
      filters = [18, 36, 72]
      expansion = 6

      with tf.variable_scope('init'):
        x = self.conv(x, 3, filters[0])

      for i in range(len(filters)):
        for j in range(num_residual):
          with tf.variable_scope('U%d-%d' % (i, j)):
            x = self._CSC(x, filters[i], stride=strides[i] if j == 0 else 1, expansion=expansion)

      with tf.variable_scope('global_avg_pool'):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], name='fc')

      return x

    # elif self.dataset in ['imagenet']:
    #
    #   with tf.variable_scope('init'):
    #     x = self.conv(x, 7, 32, stride=2)
    #     # x = self._batch_norm(x)
    #     # x = self._activation(x)
    #     # x = self._pool(x, type='MAX', ksize=3, stride=2)
    #
    #   strides = [2, 1, 2, 1, 2, 1, 2, 1]
    #   repeat = [1, 4, 1, 5, 1, 6, 1, 2]
    #   expansion = [4, 4, 4, 3, 3, 2, 2, 1]
    #   ksize = [5, 5, 5, 5, 3, 3, 3, 3]
    #   c_out = [64, 64, 128, 128, 256, 256, 512, 512]
    #
    #   for i in range(len(strides)):
    #     for j in range(repeat[i]):
    #       with tf.variable_scope('U%d-%d' % (i, j)):
    #         x = self._shift_block(x, ksize[i], expansion[i], c_out[i], strides[i])
    #
    #   with tf.variable_scope('global_avg_pool'):
    #     x = self.batch_norm(x)
    #     x = self.activation(x)
    #     x = self.pool(x, 'GLO')
    #
    #   with tf.variable_scope('logit'):
    #     x = self.fc(x, self.shape_y[1], name='fc')
    #
    #   return x


