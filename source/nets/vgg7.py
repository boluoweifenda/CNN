import tensorflow as tf
from nets.net import Net


class VGG7(Net):

  def model(self, x):
    with tf.variable_scope('U0'):
      with tf.variable_scope('C0'):
        x = self._conv(x, 3, 128)
        x = self._batch_norm(x)
        x = self._activation(x)
      with tf.variable_scope('C1'):
        x = self._conv(x, 3, 128)
        x = self._pool(x, 'MAX', 2, 2)
        x = self._batch_norm(x)
        x = self._activation(x)

    with tf.variable_scope('U1'):
      with tf.variable_scope('C0'):
        x = self._conv(x, 3, 256)
        x = self._batch_norm(x)
        x = self._activation(x)
      with tf.variable_scope('C1'):
        x = self._conv(x, 3, 256)
        x = self._pool(x, 'MAX', 2, 2)
        x = self._batch_norm(x)
        x = self._activation(x)

    with tf.variable_scope('U2'):
      with tf.variable_scope('C0'):
        x = self._conv(x, 3, 512)
        x = self._batch_norm(x)
        x = self._activation(x)
      with tf.variable_scope('C1'):
        x = self._conv(x, 3, 512)
        x = self._pool(x, 'MAX', 2, 2)
        x = self._batch_norm(x)
        x = self._activation(x)

    x = self._reshape(x)

    with tf.variable_scope('FC'):
      x = self._fc(x, 1024, name='fc0')
      x = self._batch_norm(x)
      x = self._activation(x)
      x = self._fc(x, self.shape_y[1], name='fc1')

    return x

