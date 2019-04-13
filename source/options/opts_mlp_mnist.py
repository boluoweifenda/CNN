import tensorflow as tf

title = 'temp'
mode = 'train'  # 'train', 'test', 'restart', 'debug', 'export', 'input_train', 'input_test', 'speed_net'
seed = None
repeat = 1
delay = False  # start training after xxx minutes
gpu_list = [0]
batch_size = 128
interp = 0

dataset = 'mnist'  # 'mnist','svhn','cifar10', 'cifar100', 'imagenet', 'fashion'
preprocess = 'mnist'  # 'mnist', 'cifar', 'alexnet', 'inception_v2'
network = 'mlp'  # 'mlp', 'lenet', 'alexnet', 'densenet_test', 'resnet_test', 'mobilenet_v1', 'mobilenet_v2', shufflenet_v2

path_load = None
path_save = None   # None, False, True, or specify a dir

l2_decay = {'decay': 0e-4, 'exclude': ['depthwise']}
epoch_step = tf.Variable(1, name='epoch_step', trainable=False)
learning_step = tf.Variable(0, name='learning_step', trainable=False)
lr_decay = tf.train.cosine_decay(0.1, epoch_step, decay_steps=100)
loss_func = tf.losses.softmax_cross_entropy
optimizer = tf.train.MomentumOptimizer(lr_decay, 0.9, use_nesterov=True)





