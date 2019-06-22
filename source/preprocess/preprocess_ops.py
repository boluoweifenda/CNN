from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def channel_normalize(x, mean, std):
  if isinstance(mean, list):
    mean = np.array(mean, dtype=np.float32)
  if isinstance(std, list):
    std = np.array(std, dtype=np.float32)
  # Convert division to multiplication
  std_r = 1. / std
  print('mean =', mean, 'std =', std)
  return std_r * (x - mean)


def cutout(x, h, w):

  shape = x.get_shape().as_list()
  mask = np.ones(shape=[2*shape[0], 2*shape[1]], dtype=np.float32)
  mask[(shape[0]-h//2):(shape[0]+h//2), (shape[1]-w//2):(shape[1]+w//2)] = 0
  mask = tf.constant(mask)
  mask_crop = tf.random_crop(mask, [shape[0],shape[1]])

  return x*mask_crop[:,:,None]


def central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.
  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    3-D tensor with cropped image.
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(
      image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def smallest_size_at_least(height, width, resize_min):
  """Computes new shape with the smallest side equal to `smallest_side`.
  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: an int32 scalar tensor indicating the new width.
  """
  resize_min = tf.cast(resize_min, tf.float32)

  # Convert to floats to make subsequent calculations go smoothly.
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  smaller_dim = tf.minimum(height, width)
  scale_ratio = resize_min / smaller_dim

  # Convert back to ints to make heights and widths that TF ops will accept.
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)

  return new_height, new_width


def aspect_preserving_resize(image, resize_min):
  """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  new_height, new_width = smallest_size_at_least(height, width, resize_min)

  return resize_image(image, new_height, new_width)


def resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.
  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.
  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.
  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  return tf.image.resize(
      image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)


def apply_with_random_selector(x, func, num_cases):

  sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):

  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)
