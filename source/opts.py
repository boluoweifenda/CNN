import tensorflow as tf

title = ''


# title = 'temp'
mode = 'train'  # 'train', 'test', 'restart', 'debug', 'export', 'input_train', 'input_test', 'speed_net'
seed = None
gpu_list = [3]
batch_size = 128

dataset = 'mnist'  # 'mnist', 'fashion'
preprocess = 'mnist'  # 'mnist'
network = 'mlp'  # 'mlp', 'lenet'

path_load = None
path_save = None
# path_load = '11-22'    # None, or specify a dir or key word in default dir
# path_save = True   # None, False, True, or specify a dir

l2_decay = {'decay': 0, 'exclude': []}
epoch_step = tf.Variable(1, name='epoch_step', trainable=False)
learning_step = tf.Variable(0, name='learning_step', trainable=False)
# lr_decay = tf.train.piecewise_constant(epoch_step, boundaries=[60, 80, 100], values=[1e-1, 1e-2, 1e-3, 0.0])
lr_decay = tf.train.cosine_decay(0.5, epoch_step, decay_steps=100)  # cifar cosine
# loss_func = tf.losses.softmax_cross_entropy
loss_func = tf.losses.mean_squared_error
optimizer = tf.train.MomentumOptimizer(lr_decay, momentum=0.)






