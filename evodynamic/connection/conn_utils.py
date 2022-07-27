""" Connection utils """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def weight_variable(shape, stddev=0.02, scale = 1.0, name=None):
  """
  Initializer of weights.

  Parameters
  ----------
  shape : tuple
      Shape of the weight matrix.
  stddev : float
      Standard deviation.
  name : str, optional
      Name of the Tensor.

  Returns
  -------
  out : Tensor
      Initialized connection matrix for TensorFlow.
  """
  initial = np.random.normal(scale=stddev, size=shape) * scale
  if name is None:
    return tf.Variable(initial)
  else:
    return tf.get_variable(name, initializer=initial)

def weight_variable_truncated_normal(shape, stddev=0.02, scale = 1.0, name=None):
  """
  Initializer of weights.

  Parameters
  ----------
  shape : tuple
      Shape of the weight matrix.
  stddev : float
      Standard deviation.
  name : str, optional
      Name of the Tensor.

  Returns
  -------
  out : Tensor
      Initialized connection matrix for TensorFlow.
  """
  initial = tf.multiply(
    tf.truncated_normal(shape, stddev=stddev, dtype=tf.float64),
    scale)
  if name is None:
    return tf.Variable(initial)
  else:
    return tf.get_variable(name, initializer=initial)

def weight_variable_xavier_initialized(shape, name=None):
  """
  Xavier initializer of weights.

  Parameters
  ----------
  shape : tuple
      Shape of the weight matrix.
  name : str, optional
      Name of the Tensor.

  Returns
  -------
  out : Tensor
      Initialized connection matrix for TensorFlow.

  Notes
  -----
  Based on:
  https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
  """
  in_dim = shape[1]
  xavier_stddev = 1. / np.sqrt(in_dim / 2.)
  return weight_variable(shape, stddev=xavier_stddev, name=name)