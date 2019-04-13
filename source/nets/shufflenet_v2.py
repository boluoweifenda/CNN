import tensorflow as tf
from nets.net import Net


class ShuffleNet(Net):

  def _activation(self, x):
    return tf.nn.relu6(x)

  def _channel_shuffle(self, x, num_group):
    n, c, h, w = self._shape(x)
    assert c%num_group == 0

    x_reshaped = tf.reshape(x, [-1, num_group, c//num_group, h, w])
    x_transposed = tf.transpose(x_reshaped, [0, 2, 1, 3, 4])
    x = tf.reshape(x_transposed, [-1, c, h, w])
    return x

  def _branch0(self, x, c_out, stride=1):

    if stride ==1:
      return x

    with tf.variable_scope('DW'):
      x = self._depthwise_conv(x, 3, stride=stride)
      x = self._batch_norm(x)
    with tf.variable_scope('C1'):
      x = self._conv(x, 1, c_out)
      x = self._batch_norm(x)
      x = self._activation(x)

    return x

  def _branch1(self, x, c_out, stride=1):

    with tf.variable_scope('C0'):
      x = self._conv(x, 1, c_out)
      x = self._batch_norm(x)
      x = self._activation(x)
    with tf.variable_scope('DW'):
      x = self._depthwise_conv(x, 3, stride=stride)
      x = self._batch_norm(x)
    with tf.variable_scope('C1'):
      x = self._conv(x, 1, c_out)
      x = self._batch_norm(x)
      x = self._activation(x)

    return x

  def _basic(self, x, c_out, stride=1):

    if stride == 1:
      x0, x1 = tf.split(x, num_or_size_splits=2, axis=1)
    else:
      x0 = x
      x1 = x

    c_out = int(c_out / 2)

    with tf.variable_scope('B0'):
      x0 = self._branch0(x0, c_out=c_out, stride=stride)
    with tf.variable_scope('B1'):
      x1 = self._branch1(x1, c_out=c_out, stride=stride)

    x = tf.concat([x0, x1], axis=1)
    x = self._channel_shuffle(x, num_group=2)

    return x

  def model(self, x):

    Repeat = [4, 8, 4]
    # Out = [48, 96, 192, 1024]
    # Out = [116, 232, 464, 1024]
    Out = [176, 352, 704, 1024]
    # Out = [244, 488, 976, 2048]

    with tf.variable_scope('init'):
      x = self._conv(x, 3, 24, stride=2)
      x = self._batch_norm(x)
      x = self._activation(x)
      x = self._pool(x, type='MAX', ksize=3, stride=2)

    for stage in range(len(Repeat)):
      with tf.variable_scope('S%d' % stage):
        for repeat in range(Repeat[stage]):
          with tf.variable_scope('R%d' % repeat):
            x = self._basic(x, c_out=Out[stage], stride=2 if repeat is 0 else 1)

    with tf.variable_scope('last'):
      x = self._conv(x, 1, Out[-1])
      x = self._batch_norm(x)
      x = self._activation(x)
    with tf.variable_scope('global_avg_pool'):
      x = self._pool(x, 'GLO')

    with tf.variable_scope('logit'):
      x = self._fc(x, self.shape_y[1], name='fc', bias=True)

    return x



