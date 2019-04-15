import tensorflow as tf
from nets.net import Net


class MobileNet(Net):

  def activation(self, x):
    return tf.nn.relu6(x)

  def separable_conv(self, x, ksize, c_out, stride=1, padding='SAME', name='separable_conv'):
    with tf.variable_scope(name + '_D'):
      x = self.depthwise_conv(x, ksize, channel_multiplier=1, stride=stride, padding=padding)
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope(name + '_M'):
      x = self.conv(x, 1, c_out)
      x = self.batch_norm(x)
      x = self.activation(x)
    return x

  def model(self, x):

    # 100% config
    Stride = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    Multiplier = [2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

    with tf.variable_scope('init'):
      x = self.conv(x, 3, 32, stride=2)
      x = self.batch_norm(x)
      x = self.activation(x)

    for i in range(len(Stride)):
      cout = self.get_shape(x)[1] * Multiplier[i]
      x = self.separable_conv(x, 3, cout, Stride[i], name='S%d' % i)

    with tf.variable_scope('global_avg_pool'):
      x = self.pool(x, 'GLO')

    with tf.variable_scope('logit'):
      x = self.dropout(x, 1e-3)
      x = self.fc(x, self.shape_y[1], name='fc', bias=True)

    return x



