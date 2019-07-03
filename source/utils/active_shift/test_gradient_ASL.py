from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test


from active_shift2d_op import active_shift2d
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

N = 8
Ci = 6
Co = 6
H = 4
W = 5

stride_h = 1
stride_w = 2

pad_h = 1
pad_w = 0

class DepthwiseACUTest(test.TestCase):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    def testBottomGradientFloat64(self):      
      with self.test_session(config=self.config):
          with tf.device('/gpu:0'):
              np.random.seed()
              arr = np.random.random((N, Ci, H, W))
              shift = np.random.random((2, Ci))
                
              a = tf.constant(arr, dtype=tf.float64)
              c = tf.constant(shift, dtype = np.float64)
            
              result = active_shift2d(a, c, strides=[1, 1, stride_h, stride_w], paddings=[0, 0, pad_h, pad_w])
              
              err = gradient_checker.compute_gradient_error(a, arr.shape, 
                                                            result, result.get_shape().as_list(), x_init_value=arr)
        
      print("Bottom (float64) gradient err = ", err)
      self.assertLess(err, 1e-6)
          
    def testshiftGradientFloat64(self):
      with self.test_session(config=self.config):
          with tf.device('/gpu:0'):
              np.random.seed()
              arr = np.random.random((N, Ci, H, W))
              shift = np.random.random((2, Ci))*0.5+0.2
                
              a = tf.constant(arr, dtype=tf.float64)
              c = tf.constant(shift, dtype = np.float64)
            
              result = active_shift2d(a, c, strides=[1, 1, stride_h, stride_w], paddings=[0, 0, pad_h, pad_w])
                      
              err = gradient_checker.compute_gradient_error(c, shift.shape, 
                                                            result, result.get_shape().as_list(), x_init_value=shift)
               
              #print(shift)
    
          
      print("shift (float64) gradient err = ", err)
      self.assertLess(err, 1e-6)
    

if __name__ == "__main__":  
  test.main()

