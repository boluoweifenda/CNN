import tensorflow as tf

title = 'shufflenet-v2, 2.0x'  # top1 error rate: 26.6%
mode = 'train'  # 'train', 'test', 'restart', 'debug', 'export', 'input_train', 'input_test', 'speed_net'
seed = None
repeat = 1
delay = False  # start training after xxx minutes
gpu_list = [0, 1]
batch_size = 256
data_format = 'NCHW'

dataset = 'imagenet'  # 'mnist','svhn','cifar10', 'cifar100', 'imagenet', 'fashion', 'tiny_imagenet'
preprocess = 'inception'  # 'mnist', 'cifar', 'alexnet', 'inception'
network = 'shufflenet_v2'  # 'mlp', 'lenet', 'alexnet', 'resnet', 'densenet', 'mobilenet_v1', 'mobilenet_v2', shufflenet_v2

path_load = None
path_save = True   # None, False, True, or specify a dir

l2_decay = {'decay': 0.4e-4, 'exclude': ['depthwise', 'batchnorm', 'bias']}
epoch_step = tf.Variable(1, name='epoch_step', trainable=False)
learning_step = tf.Variable(0, name='learning_step', trainable=False)
lr_decay = tf.train.cosine_decay(0.1, epoch_step, decay_steps=100)
loss_func = tf.losses.softmax_cross_entropy
optimizer = tf.train.MomentumOptimizer(lr_decay, 0.9, use_nesterov=True)

