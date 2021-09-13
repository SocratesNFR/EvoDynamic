""" Custom connections """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def create_custom_matrix(name, matrix):
  """
  Creates a custom matrix for evodynamic.connection.WeightedConnection according
  to 'matrix'.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  matrix : numpy array or list of int/float
      Matrix.

  Returns
  -------
  out : Tensor
      Converted matrix for TensorFlow.
  """
  return tf.get_variable(name, initializer=matrix)

def create_custom_sparse_matrix(name, indices, values, dense_shape):
  """
  Creates a custom sparse matrix for
  evodynamic.connection.WeightedConnection according to parameters.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  indices : list of int
      Indices of the values in the sparse Tensor.
  values : list of int/float
      Values of corresponding indices.
  dense_shape : tuple
      Shape of the dense version the sparse matrix

  Returns
  -------
  out : Tensor
      Converted sparse matrix for TensorFlow.
  """
  return tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=dense_shape), tf.float64)
