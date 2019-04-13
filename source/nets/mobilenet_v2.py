import tensorflow as tf
from nets.net import Net


class MobileNet(Net):

  def _activation(self, x):
    return tf.nn.relu6(x)

  def _expanded_conv(self, x, ksize, c_out, stride=1, expansion_size=6, residual='True', padding='SAME',
                    name='expanded_conv'):
    x_orig = x
    c_in = self._shape(x)[1]
    if expansion_size != 1:
      with tf.variable_scope(name + '_expansion'):
        x = self._conv(x, 1, c_in * expansion_size, stride=1, padding=padding)
        x = self._batch_norm(x)
        x = self._activation(x)
    with tf.variable_scope(name + '_depthwise'):
      x = self._depthwise_conv(x, ksize, channel_multiplier=1, stride=stride, padding=padding)
      x = self._batch_norm(x)
      x = self._activation(x)
    with tf.variable_scope(name + '_projection'):
      x = self._conv(x, 1, c_out, stride=1, padding=padding)
      x = self._batch_norm(x)
    if stride == 1 and residual and c_in == c_out:
      x = x + x_orig
    return x

  def model(self, x):

    Stride = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
    # Out = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64,  96,  96,  96, 160, 160, 160, 320, 1280]  # 1.0x
    Out = [48, 24, 32, 32, 48, 48, 48, 88, 88, 88, 88, 136, 136, 136, 224, 224, 224, 448, 1792]  # 1.4x
    Expansion = [1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

    with tf.variable_scope('init'):
      x = self._conv(x, 3, Out[0], stride=2)
      x = self._batch_norm(x)
      x = self._activation(x)

    for i in range(len(Stride)):
      x = self._expanded_conv(x, 3, Out[i+1], Stride[i], Expansion[i], name='block%d' % i)

    with tf.variable_scope('last'):
      x = self._conv(x, 1, Out[-1])
      x = self._batch_norm(x)
      x = self._activation(x)
    with tf.variable_scope('global_avg_pool'):
      x = self._pool(x, 'GLO')

    with tf.variable_scope('logit'):
      x = self._dropout(x, 0.2)
      x = self._fc(x, self.shape_y[1], name='fc', bias=True)

    return x



