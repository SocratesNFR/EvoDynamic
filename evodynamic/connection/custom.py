""" Custom connections """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def create_custom_matrix(name, matrix):
  return tf.get_variable(name, initializer=matrix)

def create_custom_sparse_matrix(name, indices, values, dense_shape):
  return tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=dense_shape), tf.float64)
