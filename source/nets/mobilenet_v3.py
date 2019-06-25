import tensorflow as tf
from nets.net import Net


class MobileNet(Net):

  def activation(self, x, mode=0):
    if mode == 0:
      return x
    elif mode == 1:
      return tf.nn.relu6(x)
    elif mode == 2:
      return x * tf.nn.relu6(x+3)/6

  def _expanded_conv(self, x, ksize, exp, c_out, stride, SE, NL, name='expanded_conv'):
    x_orig = x
    c_in = self.get_shape(x)[1]
    with tf.variable_scope(name + '_expansion'):
      x = self.conv(x, 1, exp, stride=1)
      x = self.batch_norm(x)
      x = self.activation(x, NL)
    with tf.variable_scope(name + '_depthwise'):
      x = self.depthwise_conv(x, ksize, channel_multiplier=1, stride=stride)
      x = self.batch_norm(x)
      x = self.activation(x, NL)
    with tf.variable_scope(name + '_projection'):
      x = self.conv(x, 1, c_out, stride=1)
      x = self.batch_norm(x)
    if stride == 1 and residual and c_in == c_out:
      x = x + x_orig
    return x

  def model(self, x):

    Ksize = [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5]
    Exp = [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960]
    Out = [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 960]
    SE = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    NL = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    Stride = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1]

    with tf.variable_scope('init'):
      x = self.conv(x, 3, 16, stride=2)
      x = self.batch_norm(x)
      x = self.activation(x, mode=2)

    for i in range(len(Stride)):
      x = self._expanded_conv(x, Ksize[i], Exp[i], Out[i], Stride[i], SE[i], NL[i], name='block%d' % i)

    with tf.variable_scope('global_avg_pool'):
      x = self.conv(x, 1, Out[-1])
      x = self.batch_norm(x)
      x = self.activation(x, mode=2)
      x = self.pool(x, 'GLO')

    with tf.variable_scope('logit'):
      x = self.dropout(x, 0.2)
      x = self.fc(x, 1280, name='fc1', bias=True)
      x = self.activation(x, mode=2)
      x = self.fc(x, self.shape_y[1], name='fc2', bias=True)

    return x



