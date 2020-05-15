""" Connection utils """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def weight_variable(shape, stddev=0.02, name=None):
  #initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float64)
  initial = np.random.normal(scale=stddev, size=shape)
  if name is None:
    return tf.Variable(initial)
  else:
    return tf.get_variable(name, initializer=initial)

def weight_variable_xavier_initialized(shape, name=None):
  # https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
  in_dim = shape[0]
  xavier_stddev = 1. / np.sqrt(in_dim / 2.)
  return weight_variable(shape, stddev=xavier_stddev, name=name)