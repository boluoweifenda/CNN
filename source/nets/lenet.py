import tensorflow as tf
from nets.net import Net


class LeNet(Net):

  def model(self, x):

    with tf.variable_scope('Conv0'):
      x = self._conv(x, 5, 32, padding='VALID', name='conv0')
      x = self._pool(x, 'MAX', 2, 2)
      x = self._activation(x)
    with tf.variable_scope('Conv1'):
      x = self._conv(x, 5, 64, padding='VALID', name='conv1')
      x = self._pool(x, 'MAX', 2, 2)
      x = self._activation(x)

    x = self._reshape(x)

    with tf.variable_scope('Fc0'):
      x = self._fc(x, 512, name='fc0')
      x = self._activation(x)
    with tf.variable_scope('Fc1'):
      x = self._fc(x, self.shape_y[1], name='fc1')

    return x



