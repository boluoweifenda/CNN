import tensorflow as tf
from nets.net import Net


class AlexNet(Net):

  def model(self, x):

    with tf.variable_scope('conv0'):
      x = self.conv(x, 11, 96, stride=4, padding='SAME', name='conv0')
      x = self.pool(x, 'MAX', 3, 2, padding='VALID')
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope('conv1'):
      x = self.conv(x, 5, 256, padding='SAME', name='conv1')
      x = self.pool(x, 'MAX', 3, 2, padding='VALID')
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope('conv2'):
      x = self.conv(x, 3, 384, padding='SAME', name='conv2')
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope('conv3'):
      x = self.conv(x, 3, 384, padding='SAME', name='conv3')
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope('conv4'):
      x = self.conv(x, 3, 256, padding='SAME', name='conv4')
      x = self.pool(x, 'MAX', 3, 2, padding='VALID')
      x = self.batch_norm(x)
      x = self.activation(x)

    x = self.reshape(x)

    with tf.variable_scope('fc0'):
      x = self.fc(x, 4096, name='fc0')
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope('fc1'):
      x = self.fc(x, 4096, name='fc1')
      x = self.batch_norm(x)
      x = self.activation(x)

      x = self.fc(x, self.shape_y[1], name='fc2')

    return x


