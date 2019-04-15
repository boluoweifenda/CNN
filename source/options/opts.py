import tensorflow as tf

title = 'temp'
mode = 'train'  # 'train', 'test', 'restart', 'debug', 'export', 'input_train', 'input_test', 'speed_net'
seed = None
repeat = 1
delay = False  # start training after xxx minutes
gpu_list = []
batch_size = 256
mixup = 0

dataset = 'mnist'  # 'mnist','svhn','cifar10', 'cifar100', 'imagenet', 'fashion'
preprocess = 'mnist'  # 'mnist', 'cifar', 'alexnet', 'inception_v2'
network = 'lenet'  # 'mlp', 'lenet', 'alexnet', 'densenet_test', 'resnet_test', 'mobilenet_v1', 'mobilenet_v2', shufflenet_v2

path_load = None
path_save = None
# path_load = '18-11-07*conv2x2*random*save'    # None, or specify a dir or key word in default dir
# path_save = True   # None, False, True, or specify a dir

l2_decay = {'decay': 1e-4, 'exclude': ['depthwise']}
# l2_decay = {'decay': 5e-4, 'exclude': ['depthwise']}  # alexnet
# l2_decay = {'decay': 0.4e-4, 'exclude': ['depthwise', 'bias', 'batchnorm']}
epoch_step = tf.Variable(1, name='epoch_step', trainable=False)
learning_step = tf.Variable(0, name='learning_step', trainable=False)
# lr_decay = tf.train.piecewise_constant(epoch_step, boundaries=[150, 225, 300], values=[1e-1, 1e-2, 1e-3, 0.0])  # resnet
# lr_decay = tf.train.piecewise_constant(epoch_step, boundaries=[60, 80, 100], values=[1e-1, 1e-2, 1e-3, 0.0])  # alexnet
# lr_decay = tf.train.piecewise_constant(epoch_step, boundaries=[10, 20], values=[1e-2, 1e-3, 0.0])  # mobilenet
# lr_decay = tf.train.cosine_decay(0.1, epoch_step, decay_steps=300)  # cifar cosine
lr_decay = tf.train.cosine_decay(0.1, epoch_step, decay_steps=600)  # cifar cosine, nasnet
# lr_decay = tf.train.cosine_decay(0.1, epoch_step, decay_steps=100)  # imagenet cosine
loss_func = tf.losses.softmax_cross_entropy
optimizer = tf.train.MomentumOptimizer(lr_decay, 0.9, use_nesterov=True)





