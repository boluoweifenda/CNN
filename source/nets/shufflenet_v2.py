import tensorflow as tf
from nets.net import Net


class ShuffleNet(Net):

  def activation(self, x):
    return tf.nn.relu6(x)

  def branch0(self, x, c_out, stride=1):

    if stride ==1:
      return x

    with tf.variable_scope('DW'):
      x = self.depthwise_conv(x, 3, stride=stride)
      x = self.batch_norm(x)
    with tf.variable_scope('C1'):
      x = self.conv(x, 1, c_out)
      x = self.batch_norm(x)
      x = self.activation(x)

    return x

  def branch1(self, x, c_out, stride=1):

    with tf.variable_scope('C0'):
      x = self.conv(x, 1, c_out)
      x = self.batch_norm(x)
      x = self.activation(x)
    with tf.variable_scope('DW'):
      x = self.depthwise_conv(x, 3, stride=stride)
      x = self.batch_norm(x)
    with tf.variable_scope('C1'):
      x = self.conv(x, 1, c_out)
      x = self.batch_norm(x)
      x = self.activation(x)

    return x

  def basic(self, x, c_out, stride=1):

    if stride == 1:
      x0, x1 = tf.split(x, num_or_size_splits=2, axis=1)
    else:
      x0 = x
      x1 = x

    c_out = int(c_out / 2)

    with tf.variable_scope('B0'):
      x0 = self.branch0(x0, c_out=c_out, stride=stride)
    with tf.variable_scope('B1'):
      x1 = self.branch1(x1, c_out=c_out, stride=stride)

    x = tf.concat([x0, x1], axis=1)
    x = self.shuffle_channel(x, num_group=2)

    return x

  def model(self, x):

    print('ShuffleNet for %s dataset' % self.dataset)

    if self.dataset in ['cifar10', 'cifar100']:

      Repeat = [9, 9, 9]
      Out = [64, 128, 256, 512]
      strides = [1, 2, 2]

      with tf.variable_scope('init'):
        x = self.conv(x, 3, Out[0])
        x = self.batch_norm(x)
        x = self.activation(x)

      for stage in range(len(Repeat)):
        with tf.variable_scope('S%d' % stage):
          for repeat in range(Repeat[stage]):
            with tf.variable_scope('R%d' % repeat):
              x = self._basic(x, c_out=Out[stage], stride=strides[stage] if repeat == 0 else 1)

      with tf.variable_scope('last'):
        x = self.conv(x, 1, Out[-1])
        x = self.batch_norm(x)
        x = self.activation(x)
      with tf.variable_scope('global_avg_pool'):
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        x = self.fc(x, self.shape_y[1], name='fc')

      return x

    elif self.dataset in ['imagenet', 'tiny_imagenet']:

      Repeat = [4, 8, 4]
      Stride = [2, 2, 2]
      # Out = [48, 96, 192, 1024]  # 0.5x
      # Out = [116, 232, 464, 1024]  # 1.0x
      # Out = [176, 352, 704, 1024]  # 1.5x
      Out = [244, 488, 976, 2048]  # 2.0x

      with tf.variable_scope('init'):

        if self.dataset == 'imagenet':
          x = self.conv(x, 3, 24, stride=2)
          x = self.batch_norm(x)
          x = self.activation(x)
          x = self.pool(x, type='MAX', ksize=3, stride=2)
        else:
          x = self.conv(x, 3, 24)
          x = self.batch_norm(x)
          x = self.activation(x)

      for stage in range(len(Repeat)):
        with tf.variable_scope('S%d' % stage):
          for repeat in range(Repeat[stage]):
            with tf.variable_scope('R%d' % repeat):
              x = self.basic(x, c_out=Out[stage], stride=Stride[stage] if repeat is 0 else 1)

      with tf.variable_scope('last'):
        x = self.conv(x, 1, Out[-1])
        x = self.batch_norm(x)
        x = self.activation(x)
      with tf.variable_scope('global_avg_pool'):
        x = self.pool(x, 'GLO')

      with tf.variable_scope('logit'):
        # dropout can improve the accuracy 27.5%->26.6%, but it is not mentioned in the paper
        # x = self.dropout(x, 0.2)
        x = self.fc(x, self.shape_y[1], name='fc', bias=True)

      return x



