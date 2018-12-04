from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce

import tensorflow as tf
import opts
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import variance_scaling_initializer
import warnings


class Net(object):
  def __init__(self, x, y, is_training=True):

    if len(opts.gpu_list) == 0:
      print('No GPU, data_format is NHWC')
      self.data_format = 'NHWC'
    else:
      print('GPU training for CNNs, data_format is NCHW')
      self.data_format = 'NCHW'

    if x.dtype is not tf.float32:
      x = tf.to_float(x)

    if self.data_format == 'NCHW':
      if self._shape(x)[1] != 1 and self._shape(x)[1] != 3:
        print('Input data format is NHWC, convert to NCHW')
        x = tf.transpose(x,[0,3,1,2])

    self.is_training = is_training
    self.shape_x = self._shape(x)
    self.shape_y = self._shape(y)


    self.H = [x]
    self.y = y
    self.learning_step = opts.learning_step


    self.gpu_list = opts.gpu_list
    self.l2_decay = opts.l2_decay
    self.loss_func = opts.loss_func

    self.W = []
    self.loss = []
    self.MACs = []
    self.initializer = variance_scaling_initializer(
      # factor=1.0, mode='FAN_IN', uniform = True,  # Caffe
      factor=2.0, mode='FAN_IN', uniform=False,  # MSRA
    )

    self.out = self.model(self.H[-1])
    self._loss(self.out, self.y)

  def model(self, x):
    raise NotImplementedError('Basic class, none network model is defined!')

  def _loss(self, out, labels):

    with tf.name_scope('loss'):
      self.loss = self.loss_func(labels, out)

    with tf.name_scope('debug'):
      self.grads_H = tf.gradients(self.loss, self.H)
      self.grads_W = tf.gradients(self.loss, self.W)

    # error calculation
    with tf.name_scope('error'):
      label = tf.argmax(labels, axis=1)
      if self.shape_y[1] > 1000:
        print('Using Top-5 error now, keep in mind that tf.nn.in_top_k has straddle problem')
        self.error = tf.reduce_mean(tf.to_float(tf.logical_not(tf.nn.in_top_k(out, label, 5))))
      else:
        self.error = tf.reduce_mean(tf.to_float(tf.not_equal(tf.argmax(out, axis=1), label)))

  def _arr(self, stride_or_ksize):
    if self.data_format == 'NCHW':
      return [1, 1, stride_or_ksize, stride_or_ksize]
    else:
      return [1, stride_or_ksize, stride_or_ksize, 1]

  def _shape(self, x):
    return x.get_shape().as_list()

  def _channel(self, x):
    shape = x.get_shape().as_list()
    if self.data_format == 'NCHW' and len(shape) == 4:
      return shape[1]
    elif self.data_format == 'NHWC' and len(shape) == 4:
      return shape[3]
    elif len(shape) == 2:
      return shape[1]
    else:
      raise NotImplementedError('Wrong shapes' + shape)


  def _activation(self, x):
    return tf.nn.relu(x)

  def _dropout(self, x, drop_prob, noise_shape=None, seed=None, name=None):
    if drop_prob > 0.00001 and self.is_training:
      x = tf.nn.dropout(x, keep_prob=1-drop_prob, noise_shape=noise_shape, seed=seed, name=name)
    return x

  def _reshape(self, x, shape=None):
    if shape is None:
      shape = [reduce(lambda x, y: x * y, self._shape(x)[1:])]
    shape = [-1] + shape
    x = tf.reshape(x, shape)
    self.H.append(x)
    return x

  def _get_variable(self, shape, name, initializer=None):
    with tf.name_scope(name) as scope:
      if initializer is None:
        initializer = self.initializer
      self.W.append(tf.get_variable(name=name, shape=shape, initializer=initializer))
    return self.W[-1]

  def _conv(self, x, ksize, c_out, stride=1, padding='SAME', bias=False, name='conv'):
    c_in = self._channel(x)
    W = self._get_variable([ksize, ksize, c_in, c_out], name)
    x = tf.nn.conv2d(x, W, self._arr(stride), padding=padding, data_format=self.data_format, name=name)
    if bias:
      b = self._get_variable([c_out], name + '_b', initializer=tf.initializers.zeros)
      x = tf.nn.bias_add(x, b, data_format=self.data_format)
    self.H.append(x)

    shape_out = self._shape(x)
    MACs = c_in*shape_out[-1]*shape_out[-2]*shape_out[-3]*ksize*ksize
    self.MACs.append([name, MACs])

    return x

  def _fc(self, x, c_out, bias=False, name='fc'):
    c_in = self._channel(x)
    W = self._get_variable([c_in, c_out], name)
    x = tf.matmul(x, W)
    if bias:
      b = self._get_variable([c_out], name+'_bias', initializer=tf.initializers.zeros)
      x = x + b
    self.H.append(x)

    MACs = c_in*c_out
    self.MACs.append([name,MACs])

    return x

  def _pool(self, x, type, ksize=2, stride=1, padding='SAME'):
    assert x.get_shape().ndims == 4, 'Invalid pooling shape:' + x.get_shape()
    if type == 'MAX':
      x = tf.nn.max_pool(x, self._arr(ksize), self._arr(stride), padding=padding, data_format=self.data_format)
    elif type == 'AVG':
      x = tf.nn.avg_pool(x, self._arr(ksize), self._arr(stride), padding=padding, data_format=self.data_format)
    elif type == 'GLO':
      x = tf.reduce_mean(x, [2, 3]) if self.data_format == 'NCHW' else tf.reduce_mean(x, [1, 2])
    else:
      raise ValueError('Invalid pooling type:' + type)
    self.H.append(x)
    return x

  def _batch_norm(self, x, center=True, scale=True, decay=0.9, epsilon=1e-5):
    x = batch_norm(x, center=center, scale=scale, is_training=self.is_training, decay=decay, epsilon=epsilon, fused=True, data_format=self.data_format)
    self.H.append(x)
    return x

  def total_parameters(self):
    dict_parameters = {}

    def dict_add(key, num):
      if key not in dict_parameters.keys():
        dict_parameters[key] = 0
      dict_parameters[key] += num

    key_list = ['batchnorm', 'instancenorm', 'conv', 'fc']

    for var in tf.trainable_variables():
      print(var.device, var.op.name, var.shape.as_list())
      name_lowcase = var.op.name.lower()
      num = reduce(lambda x, y: x * y, var.get_shape().as_list())

      has_key = False
      for key in key_list:
        if key in name_lowcase:
          dict_add(key, num)
          has_key = True
          break
      if not has_key:
        warnings.warn('Unknown parameter named ' + name_lowcase)

    total = 0
    for _,value in dict_parameters.items():
      total += value
    print('Parameters:', total, dict_parameters)
    return dict_parameters

  def total_MACs(self):
    total = 0
    for MAC in self.MACs:
      total += MAC[1]
    print('MACs:', total)
    return total

  def get_l2_loss(self):
    decay = self.l2_decay['decay']
    if decay > 0:
      list_var = []
      exclude_keys = self.l2_decay['exclude']
      list_name = []
      vars = tf.trainable_variables()
      for var in vars:
        name_lowcase = var.op.name.lower()
        exclude = False
        for e in exclude_keys:
          if e.lower() in name_lowcase:
            exclude = True
        if not exclude:
          list_var.append(tf.nn.l2_loss(var))
          list_name.append(name_lowcase)
      print('Add L2 weight decay', decay, 'to following trainable variables:')
      print(list_name)
      return decay * tf.add_n(list_var)
    else:
      print('No L2 weight decay')
      return 0.0


