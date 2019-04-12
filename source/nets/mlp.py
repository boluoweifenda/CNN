import tensorflow as tf
from nets.net import Net


class MLP(Net):

  def model(self, x):

    x = self._reshape(x)

    with tf.variable_scope('fc0'):
      x = self._fc(x, 512, name='fc0')
      x = self._activation(x)
    with tf.variable_scope('fc1'):
      x = self._fc(x, 512, name='fc1')
      x = self._activation(x)
    with tf.variable_scope('fc2'):
      x = self._fc(x, 512, name='fc2')
      x = self._activation(x)
    with tf.variable_scope('fc3'):
      x = self._fc(x, 512, name='fc3')
      x = self._activation(x)
    with tf.variable_scope('fc4'):
      x = self._fc(x, 512, name='fc4')
      x = self._activation(x)
    with tf.variable_scope('last'):
      x = self._fc(x, self.shape_y[1], name='last')

    return x


