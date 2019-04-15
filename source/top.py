from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.log import Log
from datasets.dataset_factory import get_dataset
from nets.nets_factory import get_net_fn
from utils.input_pipeline import get_batch

import numpy as np
import time
import os
import tensorflow as tf
import scipy.io as sio
from tqdm import tqdm
import glob
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--opts', '-o', default='opts', help='options file name')
args = parser.parse_args()
OPTION = args.opts


def get_cpu_id(cpu_core):
  if cpu_core is None:
    return '0-39'
  elif isinstance(cpu_core, str):
    return cpu_core
  cpu_id = ''
  for core in cpu_core:
    cpu_id = cpu_id + '%d,%d,' % (core, core + 20)
  cpu_id = cpu_id[:-1]
  return cpu_id


def get_variable(key=None):
  vars = tf.global_variables()
  vars_return = []
  for var in vars:
    if key is not None:
      if var.name.lower().find(key) > -1:
        vars_return.append(var)
    else:
      vars_return.append(var)
  return vars_return


def get_time(time_format=None):
  if time_format is not None:
    time_str = time.strftime(time_format, time.localtime())
    return time_str
  else:
    return time.time()


def set_seed(seed=None):
  if seed is None:
    seed = int(get_time())
  np.random.seed(seed)
  tf.set_random_seed(seed)
  return seed


def set_log(path):
  Log(path, 'w+', 1)  # set log file


def print_line():
  print('----------------------------------------------------------------------------------------------------')


def print_opts(path):
  print_line()
  print('Options are printed as follows:')
  lines = open(path).readlines()
  # filter out unused opts
  for line in lines:
    if line == '\n': continue
    if line[0] == '#': continue
    if line[:7] == 'import ': continue
    if line[:5] == 'from ': continue
    print(line, end=' ' if line[-2:] == '\n' else '')


def create_config_proto():
  # Build an initialization operation to run below.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.26
  config.allow_soft_placement = True
  config.log_device_placement = False
  return config


def get_session(gpu_list):

  os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(gpu) + ',' for gpu in gpu_list)
  # os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  sess = tf.InteractiveSession(config=create_config_proto())
  sess.run(tf.global_variables_initializer())
  return sess


def export(list_of_numpy, name='W'):
  mat_dict = {}
  for i in range(len(list_of_numpy)):
    mat_dict[name + '%d' % i] = list_of_numpy[i]
  sio.savemat('../model/' + name + '.mat', mat_dict)
  print('exported')


def load_model(sess, name=None):
  base_dir = '../model/'
  if name is not None:
    list = glob.glob(base_dir + '*' + name + '*.tf.*')
    assert len(list) == 3, 'Find none or more than one model file'
    path_load = list[0][:list[0].find('.tf.') + 3]
    print('Loading model from %s ...' % path_load)

    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, path_load)


def aggregate_gradients(tower_grads):
  mean_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
    grads = []
    var = grad_and_vars[0][1]
    for g, v in grad_and_vars:
      assert v == var
      if g is not None:
        grads.append(g)
    if grads:
      if len(grads) > 1:
        mean_factor = tf.constant(1. / len(grads), dtype=tf.float32)
        mean_grad = mean_factor * tf.add_n(grads, name=var.op.name + '/sum_grads')
      else:
        mean_grad = grads[0]
      mean_grads.append((mean_grad, var))
  return mean_grads


def aggregate_statistics(tower_statistics):
  if len(tower_statistics) > 1:
    return tf.reduce_mean(tower_statistics)
  else:
    return tower_statistics[0]


def delay4gpus(delay, gpu_list):
  if isinstance(delay, bool):
    if delay:
      import pynvml
      import time
      pynvml.nvmlInit()
      handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_list[0])
      while True:
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage = memory.used / memory.total
        if usage < 0.2:
          break
        else:
          print('GPU-%d is in use %.2f, still waiting' % (gpu_list[0], usage))
        time.sleep(60)
  elif isinstance(delay, int) or isinstance(delay, float):
    import time
    delay = int(delay)
    for minute in tqdm(range(delay), desc='Wait:', leave=False, smoothing=0.1):
      time.sleep(60)
  else:
    raise NotImplementedError('Wrong delay type')


def main():

  ####################################################################################################

  title = opts.title
  seed = opts.seed
  mode = opts.mode

  gpu_list = opts.gpu_list
  batch_size = opts.batch_size

  dataset = opts.dataset
  preprocess = opts.preprocess
  network = opts.network
  optimizer = opts.optimizer
  lr_decay = opts.lr_decay
  epoch_step = opts.epoch_step
  learning_step = opts.learning_step

  path_load = opts.path_load
  path_save = opts.path_save

  print_line()

  ####################################################################################################

  time_tag = get_time('%y-%m-%d %X')
  time_tag_short = time_tag[:8]
  seed = set_seed(seed)

  num_check_log = 0
  title_temp = title
  while True:
    path_log = '../log/' + time_tag_short + '(' + title_temp + ').txt'
    if os.path.isfile(path_log) and title != 'temp':  # if title is 'temp', we will overwrite it
      num_check_log += 1
      title_temp = title + '_%d' % num_check_log
    else:
      title = title_temp
      del num_check_log, title_temp
      break

  print('title: ' + title)
  set_log(path_log)
  print_line()

  ####################################################################################################

  print(time_tag)
  print('SEED = %d' % seed)

  print_opts('options/' + OPTION + '.py')
  print_line()

  ####################################################################################################

  if isinstance(path_save, bool):
    # if title is 'temp', we will not save model
    path_save = '../model/' + time_tag_short + '(' + title + ').tf' if path_save and title != 'temp' else None

  if path_load is not None:
    # key word search
    list = glob.glob('../model/*' + path_load + '*.tf.data*')
    assert len(list) == 1, 'Find none or more than one model file'
    path_load = list[0][:list[0].find('.tf.') + 3]
    print('Find model in', path_load)

  ####################################################################################################

  num_worker = max(len(gpu_list), 1)
  dataset_train = get_dataset(dataset, split='train')
  dataset_test = get_dataset(dataset, split='test')

  num_batch_train = dataset_train.num_sample // batch_size
  num_batch_test = dataset_test.num_sample // 100

  assert batch_size % num_worker == 0, 'batch_size %d can not be divided by number of workers %d' % (batch_size, num_worker)

  iterator_train = get_batch(dataset_train, preprocess, True, batch_size // num_worker, num_worker, seed=seed)
  iterator_test = get_batch(dataset_test, preprocess, False, 100, num_worker, seed=seed)

  ####################################################################################################

  if mode in ['input_train', 'input_test']:
    if mode == 'input_train':
      num_batch = num_batch_train
      batch_input = iterator_train.get_next()
    else:
      num_batch = num_batch_test
      batch_input = iterator_test.get_next()

    sess = get_session(gpu_list)

    print('Testing the speed of data input pipeline.')

    while True:
      for batch in tqdm(range(num_batch), desc='Input pipeline', leave=False, smoothing=0.1):
        batch_input_ = sess.run(batch_input)

  ####################################################################################################

  nets = []
  net = get_net_fn(network)

  if num_worker == 1:
    if len(gpu_list) == 0:
      print('Multi-CPU training, it might be slow', )
      print('All parameters are pinned to CPU, all Ops are pinned to CPU')
      is_cpu_ps = True
    else:
      print('Single-GPU training with gpu', gpu_list[0])
      print('All parameters are pinned to GPU, all Ops are pinned to GPU')
      is_cpu_ps = False

  elif num_worker > 1:
    print('Multi-GPU training tower with gpu list', gpu_list)
    print('All parameters are pinned to CPU, all Ops are pinned to GPU')
    print('Get batchnorm moving average updates from data in the first GPU for speed')
    print('Get L2 decay grads in the second GPU for speed')
    is_cpu_ps = True
  else:
    raise NotImplementedError('Unrecognized device settings')

  tower_grads = []
  tower_losses = []
  tower_errors = []

  # Loops over the number of workers and creates a copy ("tower") of the model on each worker.
  for i in range(num_worker):

    worker = '/gpu:%d' % i if gpu_list else '/cpu:0'

    # Creates a device setter used to determine where Ops are to be placed.
    if is_cpu_ps:
      # tf.train.replica_device_setter supports placing variables on the CPU, all
      # on one GPU, or on ps_servers defined in a cluster_spec.
      device_setter = tf.train.replica_device_setter(worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
    else:
      device_setter = worker

    '''
    1. pin ops to GPU
    2. pin parameters to CPU (multi-GPU training) or GPU (single-GPU training)
    3. reuse parameters multi-GPU training

    # Creates variables on the first loop.  On subsequent loops reuse is set
    # to True, which results in the "towers" sharing variables.
    # tf.device calls the device_setter for each Op that is created.
    # device_setter returns the device the Op is to be placed on.
    '''
    with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)), \
         tf.device(device_setter):

      print('Training model on GPU %d' % gpu_list[i]) if gpu_list else print('Training model on CPUs')

      batch_train = iterator_train.get_next()

      if mode == 'speed_net':
        with tf.device('/cpu:0'):
          print('Testing the speed of model by synthesized data, '
                'which is theoretically the maximum speed for training this model')
          batch_train = iterator_train.get_next()
          shape_x = [batch_size // num_worker] + batch_train[0].get_shape().as_list()[1:]
          shape_y = [batch_size // num_worker] + batch_train[1].get_shape().as_list()[1:]

          batch_train_x = tf.zeros(shape_x, dtype=tf.float32)
          batch_train_y = tf.zeros(shape_y, dtype=tf.float32)
        batch_train = [batch_train_x, batch_train_y]

      nets.append(net(batch_train[0], batch_train[1], opts=opts, is_training=True))

      tower_losses.append(nets[i].loss)
      tower_errors.append(nets[i].error)

      if i == 0:
        print('We only get batchnorm moving average updates from data in the first worker for speed')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        nets[-1].count_parameters()
        nets[-1].count_MACs()
        nets[-1].count_MEMs()

      loss_worker = nets[i].loss
      if num_worker == 1:
        # Single-GPU or multi-CPU training
        loss_worker += nets[i].get_l2_loss()
      elif i == 1:
        # We only compute L2 grads in the second worker for speed.
        # In this case, L2 grads should multiple num_worker to maintain the equivalence
        loss_worker += num_worker * nets[i].get_l2_loss()
      tower_grads.append(
        optimizer.compute_gradients(loss_worker, colocate_gradients_with_ops=True))

      if i == num_worker - 1:
        print('Testing model on GPU %d' % gpu_list[i]) if gpu_list else print('Testing model on CPUs')
        if num_worker == 1:
          tf.get_variable_scope().reuse_variables()

        batch_test = iterator_test.get_next()
        nets.append(net(batch_test[0], batch_test[1], opts=opts, is_training=False))
        error_batch_test = nets[-1].error

  with tf.device('/cpu:0' if is_cpu_ps else worker):
    grad_batch_train = aggregate_gradients(tower_grads)
    loss_batch_train = aggregate_statistics(tower_losses)
    error_batch_train = aggregate_statistics(tower_errors)

    with tf.control_dependencies(update_ops):
      train_op = optimizer.apply_gradients(grad_batch_train, global_step=learning_step)

  ####################################################################################################

  if hasattr(opts, 'delay'):
    delay4gpus(opts.delay, gpu_list=gpu_list)

  sess = get_session(gpu_list)
  saver = tf.train.Saver(max_to_keep=None)

  def evaluate():
    error_test = 0.
    for _ in tqdm(range(num_batch_test), desc='Test', leave=False, smoothing=0.1):
      error_test += sess.run([error_batch_test])[0]
    return error_test / num_batch_test

  def load_model(path):
    print('Loading model from', path)
    saver.restore(sess, path)

  def save_model(path):
    saver.save(sess, path)
    print('S', end='')

  if path_load is not None:
    load_model(path_load)
    error_test_best = evaluate()
    print('Test: %.4f' % error_test_best)

  if mode == 'test':
    error_test_best = evaluate()
    print('Test: %.4f' % error_test_best)
    exit(0)

  if mode == 'export':
    vars_list = get_variable('batchnorm/gamma:')
    vars_numpy = sess.run(vars_list)
    export(vars_numpy, 'gamma')
    exit(0)

  if mode == 'restart':
    sess.run(epoch_step.assign(90))

  print_line()

  ####################################################################################################

  while True:
    # update learning rate
    lr_epoch = sess.run(lr_decay)
    if lr_epoch <= 0:
      break
      # sess.run(epoch_step.assign(1))
    epoch = sess.run(epoch_step)
    print('Epoch: %03d' % epoch, end=' ')

    loss_epoch = 0.
    error_epoch = 0.
    t0 = get_time()
    for batch in tqdm(range(num_batch_train), desc='Epoch: %03d' % epoch, leave=False, smoothing=0.1):

      if mode == 'debug':
        print('DEBUG: '),
        _, loss_delta, error_delta, H, W, gradsH, gradsW, label_ = sess.run(
          [train_op, loss_batch_train, error_batch_train, nets[0].H, nets[0].W, nets[0].grads_H, nets[0].grads_W,
           nets[0].y])
      else:
        _, loss_delta, error_delta = sess.run([train_op, loss_batch_train, error_batch_train])

      loss_epoch += loss_delta
      error_epoch += error_delta

    print('Loss: %.6f Train: %.4f' % (loss_epoch / num_batch_train, error_epoch / num_batch_train), end=' ')
    FPS = num_batch_train * batch_size / (get_time() - t0)

    error_test = evaluate()
    assert error_test > 1e-4, ('Invalid test error %f, something goes wrong' % error_test)
    print('Test: %.4f lr: %.4f FPS: %d' % (error_test, lr_epoch, FPS), end=' ')

    sess.run(epoch_step.assign(epoch + 1))

    if epoch == 1:
      error_test_best = min(error_test, 0.9)
    if error_test < error_test_best:
      print('B', end=' ')
      if path_save is not None:
        save_model(path_save)
      error_test_best = error_test

    print('')

  print_line()

  ####################################################################################################

  sess.close()
  print('Optimization ended at ' + get_time('%y-%m-%d %X'))
  return 0

  ####################################################################################################


if __name__ == '__main__':

  opts = importlib.import_module('options.' + OPTION)
  repeat = opts.repeat
  if repeat > 1:
    print('multiple runs, DO NOT EDIT the option file until the last run starts !!!')
  for i in range(repeat):
    tf.reset_default_graph()
    importlib.reload(opts)
    with tf.device('/cpu:0'):
      main()
  exit(0)









