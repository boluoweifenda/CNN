from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import instance_norm
from tensorflow.contrib.layers import variance_scaling_initializer
from preprocess.preprocess_ops import channel_normalize
import warnings

DATA_FORMAT = 'NCHW'


class Net(object):
  def __init__(self, x, y, opts, is_training=True):

    print('Perform input channel normalization in GPU for speed')

    dataset = opts.dataset
    preprocess = opts.preprocess

    if x.dtype is not tf.float32:
      warnings.warn('input datatype for network is not float32, please check the preprocessing')
      x = tf.cast(x, dtype=tf.float32)

    if dataset in ['imagenet', 'imagenet256']:
      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]
      if preprocess == 'alexnet':
        x = channel_normalize(x, 255*np.array(mean), 255*np.array(std))
      elif preprocess in ['inception', 'inception_v2']:
        print('In the %s argumentation, image are already scaled to [0,1]' % preprocess)
        x = channel_normalize(x, mean, std)
      else:
        raise NotImplementedError('No normalization for dataset %s and preprocess %s' % (dataset,preprocess))
    else:
      print('No normalization in GPU for dataset %s' % dataset)

    if DATA_FORMAT is 'NCHW':
      print('Input data format is NHWC, convert to NCHW')
      x = tf.transpose(x,[0,3,1,2])

    self.is_training = is_training
    self.shape_x = self._shape(x)
    self.shape_y = self._shape(y)

    x, y = self._mixup(x, y, alpha=opts.mixup)

    self.H = [x]
    self.collect = []
    self.y = y
    self.learning_step = opts.learning_step
    self.batch_size = opts.batch_size
    self.gpu_list = opts.gpu_list
    self.l2_decay = opts.l2_decay
    self.loss_func = opts.loss_func

    self.W = []
    self.loss = []
    self.MACs = []
    self.MEMs = []
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
        self.error = tf.reduce_mean(tf.cast(tf.logical_not(tf.nn.in_top_k(out, label, 5)), dtype=tf.float32))
      else:
        self.error = tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(out, axis=1), label), dtype=tf.float32))

  def _arr(self, stride_or_ksize, data_format=DATA_FORMAT):
    if data_format is 'NCHW':
      return [1, 1, stride_or_ksize, stride_or_ksize]
    else:
      return [1, stride_or_ksize, stride_or_ksize, 1]

  def _shape(self, x):
    return x.get_shape().as_list()

  def _mul_all(self, mul_list):
    return reduce(lambda x, y: x * y, mul_list)

  def _activation(self, x):
    return tf.nn.relu(x)

  def _dropout(self, x, drop_prob, noise_shape=None, seed=None, name=None):
    if drop_prob > 0.00001 and self.is_training:
      x = tf.nn.dropout(x, keep_prob=1-drop_prob, noise_shape=noise_shape, seed=seed, name=name)
    return x

  def _reshape(self, x, shape=None):
    if shape is None:
      shape = [self._mul_all(self._shape(x)[1:])]
    shape = [-1] + shape
    x = tf.reshape(x, shape)
    self.H.append(x)
    return x

  def _get_variable(self, shape, name, initializer=None):
    with tf.name_scope(name):
      if initializer is None:
        initializer = self.initializer
      self.W.append(tf.get_variable(name=name, shape=shape, initializer=initializer))
    return self.W[-1]

  def _conv(self, x, ksize, c_out, stride=1, padding='SAME', bias=False, data_format=DATA_FORMAT, name='conv'):
    shape_in = self._shape(x)
    c_in = shape_in[1] if data_format is 'NCHW' else shape_in[-1]
    W = self._get_variable([ksize, ksize, c_in, c_out], name)
    x = tf.nn.conv2d(x, W, self._arr(stride, data_format=data_format), padding=padding, data_format=data_format, name=name)
    if bias:
      b = self._get_variable([c_out], name + '_b', initializer=tf.initializers.zeros)
      x = tf.nn.bias_add(x, b, data_format=data_format)
    self.H.append(x)

    shape_out = self._shape(x)
    MEMs = self._mul_all(shape_out[1:])
    MACs = c_in*ksize*ksize*MEMs
    self.MACs.append([name, MACs])
    self.MEMs.append([name, MEMs])

    return x

  def _depthwise_conv(self, x, ksize, channel_multiplier=1, stride=1, padding='SAME', data_format=DATA_FORMAT, name='depthwise_conv'):
    shape_in = self._shape(x)
    c_in = shape_in[1] if data_format is 'NCHW' else shape_in[-1]

    initializer = variance_scaling_initializer(factor=2.0, mode='FAN_OUT', uniform=False)  # MSRA

    W = self._get_variable([ksize, ksize, c_in, channel_multiplier], name, initializer)
    x = tf.nn.depthwise_conv2d(x, W, self._arr(stride, data_format=data_format), padding=padding, data_format=data_format, name=name)
    self.H.append(x)

    shape_out = self._shape(x)
    MEMs = self._mul_all(shape_out[1:])
    MACs = ksize*ksize*MEMs
    self.MACs.append([name,MACs])
    self.MEMs.append([name, MEMs])

    return x

  def _channel_shuffle(self, x, num_group, data_format=DATA_FORMAT):
    if data_format is 'NCHW':
      n, c, h, w = self._shape(x)
      assert c % num_group == 0
      x_reshaped = tf.reshape(x, [-1, num_group, c // num_group, h, w])
      x_transposed = tf.transpose(x_reshaped, [0, 2, 1, 3, 4])
      out = tf.reshape(x_transposed, [-1, c, h, w])
    else:
      n, h, w, c = self._shape(x)
      assert c % num_group == 0
      x_reshaped = tf.reshape(x, [-1, h, w, num_group, c // num_group])
      x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
      out = tf.reshape(x_transposed, [-1, h, w, c])

    return out

  def _mixup(self, x, y, alpha):
    if 0 < alpha <= 1 and self.is_training:
      print('use mixup input data augmentation, alpha=' % alpha)
      random = tf.random_uniform([tf.shape(x)[0], 1, 1, 1], minval=0, maxval=alpha, dtype=tf.float32)
      x_slide = tf.concat([x[1:, ...], x[0:1, ...]], axis=0)
      y_slide = tf.concat([y[1:, ...], y[0:1, ...]], axis=0)
      x = random * x + (1 - random) * x_slide
      random_squeeze = random[:, 0:, 0, 0]
      y = random_squeeze * y + (1 - random_squeeze) * y_slide
    return x, y

  def _group_conv(self, x, ksize, c_out, num_group=None, stride=1, padding='SAME', shuffle=False, data_format=DATA_FORMAT, name='group_conv'):
    shape_in = self._shape(x)
    c_in = shape_in[1] if data_format is 'NCHW' else shape_in[-1]
    assert c_in % num_group == 0 and c_out % num_group == 0
    c_out_group = c_out//num_group

    initializer0 = self.initializer

    self.initializer = variance_scaling_initializer(factor=2.0 / num_group, mode='FAN_IN', uniform=False)

    axis = 1 if data_format is 'NCHW' else -1
    X = tf.split(x, num_group, axis=axis)
    Y = []
    for i in range(num_group):
      Y.append(self._conv(X[i], ksize, c_out_group, stride, padding, name=name + '-%d'%i))
    x = tf.concat(Y, axis=axis)

    if shuffle:
      x = self._channel_shuffle(x, num_group=num_group, data_format=data_format)

    self.H.append(x)
    self.initializer = initializer0

    shape_out = self._shape(x)
    MEMs = self._mul_all(shape_out[1:])
    MACs = c_in*ksize*ksize*MEMs // num_group
    self.MACs.append([name, MACs])
    self.MEMs.append([name, MEMs])

    return x

  def _separable_conv(self, x, ksize, c_out, stride=1, padding='SAME', name='separable_conv'):
    c_in = self._shape(x)[1]

    initializer = variance_scaling_initializer(
      factor=2.0, mode='FAN_OUT', uniform=False,  # MSRA
    )

    depthwise_filter = self._get_variable([ksize, ksize, c_in, 1], name + '_d', initializer)
    pointwise_filter = self._get_variable([1, 1, c_in, c_out], name + '_p', self.initializer)
    x = tf.nn.separable_conv2d(x, depthwise_filter=depthwise_filter, pointwise_filter=pointwise_filter,
                               strides=self._arr(stride), padding=padding, name=name, data_format='NCHW')

    shape_out = self._shape(x)
    MACs = shape_out[-1]*shape_out[-2]*c_in*c_out + shape_out[-1]*shape_out[-2]*c_in*ksize*ksize
    self.MACs.append([name,MACs])

    MEMs = shape_out[-1]*shape_out[-2]*shape_out[-3]*2
    self.MEMs.append([name, MEMs])

    return x

  def _conv_t(self, x, ksize, c_out, stride=1, padding='SAME', bias=False, name='conv_t'):
    shape = self._shape(x)
    from tensorflow.python.layers import utils
    out_H = utils.deconv_output_length(shape[2], ksize, padding.lower(), stride)
    out_W = utils.deconv_output_length(shape[3], ksize, padding.lower(), stride)
    output_shape = [shape[0], c_out, out_H, out_W]
    W = self._get_variable([ksize, ksize, c_out, shape[1]], name)
    x = tf.nn.conv2d_transpose(x, W, output_shape, self._arr(stride), padding=padding, data_format='NCHW', name=name)
    if bias:
      b = self._get_variable([c_out], name + '_b', initializer=tf.initializers.zeros)
      x = tf.nn.bias_add(x, b, data_format='NCHW')
    self.H.append(x)
    return x

  def _fc(self, x, c_out, bias=False, name='fc'):
    c_in = self._shape(x)[1]
    W = self._get_variable([c_in, c_out], name)
    x = tf.matmul(x, W)
    if bias:
      b = self._get_variable([c_out], name+'_bias', initializer=tf.initializers.zeros)
      x = x + b
    self.H.append(x)

    MACs = c_in*c_out
    self.MACs.append([name,MACs])

    MEMs = c_out
    self.MEMs.append([name, MEMs])

    return x

  def _scale(self, x, name='scale', data_format='NCHW'):
    shape = self._shape(x)
    if len(shape) == 4:
      if data_format == 'NCHW':
        shape = [1, shape[1], 1, 1]
      else:
        shape = [1, 1, 1, shape[3]]
    else:
      shape = shape[-1]
    scale = self._get_variable(shape, name, initializer=tf.initializers.ones)
    x = scale * x
    return x

  def _bias(self, x, name='bias', data_format='NCHW'):
    shape = self._shape(x)
    if len(shape) == 4:
      if data_format == 'NCHW':
        c_out = shape[1]
      else:
        c_out = shape[3]
    else:
      c_out = shape[-1]
    bias = self._get_variable([c_out], name, initializer=tf.initializers.zeros)
    x = tf.nn.bias_add(x, bias, data_format=data_format)
    return x

  def _linear(self, x, name='linear', data_format='NCHW'):
    x = self._scale(x, name=name+'_s', data_format=data_format)
    x = self._bias(x, name=name + '_b', data_format=data_format)
    return x

  def _pool(self, x, type, ksize=2, stride=1, padding='SAME', data_format='NCHW'):
    assert x.get_shape().ndims == 4, 'Invalid pooling shape:' + x.get_shape()
    if type == 'MAX':
      x = tf.nn.max_pool(x, self._arr(ksize), self._arr(stride), padding=padding, data_format=data_format)
    elif type == 'AVG':
      x = tf.nn.avg_pool(x, self._arr(ksize), self._arr(stride), padding=padding, data_format=data_format)
    elif type == 'GLO':
      x = tf.reduce_mean(x, [2, 3]) if data_format == 'NCHW' else tf.reduce_mean(x, [1, 2])
    else:
      raise ValueError('Invalid pooling type:' + type)
    self.H.append(x)
    return x

  def _batch_norm(self, x, center=True, scale=True, decay=0.9, epsilon=1e-5, data_format='NCHW'):
    x = batch_norm(x, center=center, scale=scale, is_training=self.is_training, decay=decay, epsilon=epsilon, fused=True, data_format=data_format)
    self.H.append(x)

    shape_out = self._shape(x)

    MEMs = shape_out[-1]*shape_out[-2]*shape_out[-3]
    self.MEMs.append(['batchnorm', MEMs])

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

  def total_MEMs(self):
    total = 0
    for MEM in self.MEMs:
      total += MEM[1]
    total = total * self.batch_size * 4 // (1024*1024)
    print('MEMs:', total)
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


