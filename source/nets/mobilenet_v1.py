import tensorflow as tf
from nets.net import Net


class MobileNet(Net):

  def _activation(self, x):
    return tf.nn.relu6(x)

  def _separable_conv(self, x, ksize, c_out, stride=1, padding='SAME', name='separable_conv'):
    with tf.variable_scope(name + '_D'):
      x = self._depthwise_conv(x, ksize, channel_multiplier=1, stride=stride, padding=padding)
      x = self._batch_norm(x)
      x = self._activation(x)
    with tf.variable_scope(name + '_M'):
      x = self._conv(x, 1, c_out)
      x = self._batch_norm(x)
      x = self._activation(x)
    return x

  def model(self, x):

    # 100% config
    Stride = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    Multiplier = [2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

    with tf.variable_scope('init'):
      x = self._conv(x, 3, 32, stride=2)
      x = self._batch_norm(x)
      x = self._activation(x)

    for i in range(len(Stride)):
      cout = self._shape(x)[1] * Multiplier[i]
      x = self._separable_conv(x, 3, cout, Stride[i], name='S%d' % i)

    with tf.variable_scope('global_avg_pool'):
      x = self._pool(x, 'GLO')

    with tf.variable_scope('logit'):
      x = self._dropout(x, 1e-3)
      x = self._fc(x, self.shape_y[1], name='fc', bias=True)

    return x



