import tensorflow as tf

# title = 'Wide-DenseNet BC, C10, K32, depthwise, no relu after conv1x1'
# title = 'shufflenet_v2, 1.5x, lr 0.1, relu6'
# title = 'vgg7, batchsize 64, epoch 300'
# title = 'ResNet-74, C10, conv4x4, sp'
# title = 'DenseNet C100 L94 K12, conv2x2'
# title = 'mobilenet-v2, 1.4x, lr 0.1'
# title = 'ResNet-50, 0.5x, L2 4e-5, lr 0.1, conv2x2, remove conv2x2 L2'
# title = 'shufflenet-v2, conv2x2_sp BC, relu after conv2x2'
# title = 'Wide-DenseNet, BC, 988, K48, 600 epoch, conv2x2, normalize first, cutout'
# title = 'DenseNet121, cosine, inception, conv3x3'



title = 'temp'
mode = 'speed_net'  # 'train', 'test', 'restart', 'debug', 'export', 'input_train', 'input_test', 'speed_net'
seed = None
# repeat = 3
delay = False  # start training after xxx minutes
gpu_list = [0,1]
batch_size = 256
# interp = 0.5

dataset = 'imagenet'  # 'mnist','svhn','cifar10', 'cifar100', 'imagenet', 'fashion'
preprocess = 'inception_v2'  # 'mnist', 'cifar', 'alexnet', 'inception_v2'
network = 'mobilenet_v2'  # 'mlp', 'lenet', 'alexnet', 'densenet_test', 'resnet_test', 'mobilenet_v1', 'mobilenet_v2', shufflenet_v2

path_load = None
# path_save = None
# path_load = '18-11-07*conv2x2*random*save'    # None, or specify a dir or key word in default dir
path_save = True   # None, False, True, or specify a dir

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





