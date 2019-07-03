import numpy as np
import tensorflow as tf
import lib.active_shift2d_op as active_shift2d_op
import os

gpu_list = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ''.join(str(gpu) + ',' for gpu in gpu_list)


def create_config_proto():
  # Build an initialization operation to run below.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.26
  config.allow_soft_placement = True
  config.log_device_placement = False
  return config


fm_np = np.reshape(np.arange(start=1, stop=17), (1,1,4,4))
kernel_np = np.array([[1, 0.5]])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=create_config_proto())
sess.run(tf.global_variables_initializer())


fm = tf.constant(fm_np, dtype=tf.float32)
kernel = tf.constant(kernel_np, dtype=np.float32)
result = active_shift2d_op.active_shift2d_op(fm, kernel, strides=[1, 1, 1, 1], paddings=[0, 0, 0, 0])
sm = sess.run(result)
grad = tf.gradients(result, [fm, kernel])
res = [sess.run(g) for g in grad]

# print(res[0])
print(sm)

