""" Connections for reservoir neural networks """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from . import conn_utils

# Based on https://github.com/nschaetti/EchoTorch/blob/master/echotorch/nn/ESNCell.py

def create_gaussian_matrix(name, width, mean=0.0, std=1.0, sparsity=None, is_sparse=False):
  """
  Creates a random square matrix with Gaussian distribution according to
  parameters for evodynamic.connection.WeightedConnection.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  width : int
      Width of the adjacency matrix.
  mean : float
      Mean for the Gaussian distribution.
  std : float
      Standard deviation for the Gaussian distribution.
  sparsity : float between 0 and 1
      Percentage of zeros in the matrix.
  is_sparse : Boolean
      Determines whether the returning Tensor is sparse or not.

  Returns
  -------
  out : Tensor
      Random adjacency matrix for TensorFlow.
  """
  nodes = width
  size = (width, width)
  if is_sparse:
    indices = []
    values = []
    if sparsity is None:
      values = np.random.normal(loc=mean, scale=std, size=nodes*nodes)
      for i in range(nodes):
        for j in range(nodes):
          indices.append([i,j])
    else:
      for i in range(nodes):
        for j in range(nodes):
          if sparsity < np.random.random():
            indices.append([i,j])
            values.append(np.random.normal(loc=mean, scale=std))
    initial = tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=[nodes, nodes]), tf.float64)
  else:
    if sparsity is None:
      conn_matrix = np.random.normal(loc=mean, scale=std, size=size)
    else:
      conn_matrix = np.zeros(size)
      for i in range(nodes):
        for j in range(nodes):
          if sparsity < np.random.random():
            conn_matrix[i,j] = np.random.normal(loc=mean, scale=std)
    initial = conn_matrix

  return initial if is_sparse else tf.get_variable(name, initializer=initial)

def create_uniform_matrix(name, width, sparsity=None, is_sparse=False):
  """
  Creates a random square matrix with Uniform distribution of weights
  according to parameters for evodynamic.connection.WeightedConnection.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  width : int
      Width of the adjacency matrix.
  sparsity : float between 0 and 1
      Percentage of zeros in the matrix.
  is_sparse : Boolean
      Determines whether the returning Tensor is sparse or not.

  Returns
  -------
  out : Tensor
      Random adjacency matrix for TensorFlow.
  """
  nodes = width
  size = (width, width)
  if is_sparse:
    indices = []
    values = []
    if sparsity is None:
      values = np.random.randint(2, size=nodes*nodes) * 2.0 - 1.0
      for i in range(nodes):
        for j in range(nodes):
          indices.append([i,j])
    else:
      for i in range(nodes):
        for j in range(nodes):
          if sparsity < np.random.random():
            indices.append([i,j])
            values.append(np.random.randint(2) * 2.0 - 1.0)

    initial = tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=[nodes, nodes]), tf.float64)
  else:
    if sparsity is None:
      conn_matrix = np.random.randint(2, size=size) * 2.0 - 1.0
    else:
      conn_matrix = np.zeros(size)
      for i in range(nodes):
        for j in range(nodes):
          if sparsity < np.random.random():
            conn_matrix[i,j] = np.random.randint(2) * 2.0 - 1.0
    initial = conn_matrix

  return initial if is_sparse else tf.get_variable(name, initializer=initial)

def create_esn_matrix(name, width, mean_pos=0.0, std_pos=1.0,\
                      mean_neg=0.0, std_neg=1.0, pos_neg_prop=0.5,\
                      sparsity=None, is_sparse=False):
  """
  Creates a random square matrix for an echo state network to be used
  by evodynamic.connection.WeightedConnection'.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  width : int
      Width of the adjacency matrix.
  mean_pos : float
      Mean for the Gaussian distribution of positive weights.
  std_pos : float
      Standard deviation for the Gaussian distribution of positive weights.
  mean_neg : float
      Mean for the Gaussian distribution of negative weights.
  std_neg : float
      Standard deviation for the Gaussian distribution of negative weights.
  pos_neg_prop : float between 0 and 1
      Proportion of positive weights over negative ones.
  sparsity : float between 0 and 1
      Percentage of zeros in the matrix.
  is_sparse : Boolean
      Determines whether the returning Tensor is sparse or not.

  Returns
  -------
  out : Tensor
      Random adjacency matrix for TensorFlow.
  """
  nodes = width
  size = (width, width)
  if is_sparse:
    indices = []
    values = []
    if sparsity is None:
      values_pos = np.abs(np.random.normal(loc=mean_pos, scale=std_pos, size=nodes*nodes))
      values_neg = -np.abs(np.random.normal(loc=mean_neg, scale=std_neg, size=nodes*nodes))
      values = np.where(np.random.random(size=nodes*nodes)<=pos_neg_prop,\
                        values_pos, values_neg)
      for i in range(nodes):
        for j in range(nodes):
          indices.append([i,j])
    else:
      for i in range(nodes):
        for j in range(nodes):
          if sparsity < np.random.random():
            if np.random.random()<=pos_neg_prop:
              indices.append([i,j])
              values.append(np.abs(np.random.normal(loc=mean_pos, scale=std_pos)))
            else:
              indices.append([i,j])
              values.append(-np.abs(np.random.normal(loc=mean_neg, scale=std_neg)))

    initial = tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=[nodes, nodes]), tf.float64)
  else:
    if sparsity is None:
      conn_matrix_pos = np.abs(np.random.normal(loc=mean_pos, scale=std_pos, size=size))
      conn_matrix_neg = -np.abs(np.random.normal(loc=mean_neg, scale=std_neg, size=size))
      conn_matrix = np.where(np.random.random(size=nodes*nodes)<=pos_neg_prop,\
                             conn_matrix_pos, conn_matrix_neg)
    else:
      conn_matrix = np.zeros(size)
      for i in range(nodes):
        for j in range(nodes):
          if sparsity < np.random.random():
            if np.random.random()<=pos_neg_prop:
              conn_matrix[i,j] = np.abs(np.random.normal(loc=mean_pos, scale=std_pos))
            else:
              conn_matrix[i,j] = -np.abs(np.random.normal(loc=mean_neg, scale=std_neg))
    initial = conn_matrix

  return initial if is_sparse else tf.get_variable(name, initializer=initial)

def create_xavier_connection(name, from_group_amount, to_group_amount):
  """
  Xavier initializer of a connection.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  from_group_amount : int
      Number of cells in the 'from_group'.
  to_group_amount : int
      Number of cells in the 'to_group'.

  Returns
  -------
  out : Tensor
      Random adjacency matrix for TensorFlow.
  """
  return conn_utils.weight_variable_xavier_initialized([to_group_amount, from_group_amount], name=name)

def create_normal_distribution_connection(name, from_group_amount, to_group_amount, stddev=0.02):
  """
  Normal distribution initializer of a connection.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  from_group_amount : int
      Number of cells in the 'from_group'.
  to_group_amount : int
      Number of cells in the 'to_group'.
  stddev : int
      Standard deviation of the normal distribution (mean=0.0).

  Returns
  -------
  out : Tensor
      Random adjacency matrix for TensorFlow.
  """
  return conn_utils.weight_variable([to_group_amount, from_group_amount],
                                    stddev=stddev, name=name)

def create_truncated_normal_connection(name, from_group_amount, to_group_amount, stddev=0.02):
  """
  Truncated normal initializer of a connection.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  from_group_amount : int
      Number of cells in the 'from_group'.
  to_group_amount : int
      Number of cells in the 'to_group'.
  stddev : int
      Standard deviation of the normal distribution (mean=0.0).

  Returns
  -------
  out : Tensor
      Random adjacency matrix for TensorFlow.
  """
  return conn_utils.weight_variable_truncated_normal([to_group_amount, from_group_amount],
                                    stddev=stddev, name=name)

def create_uniform_connection(name, from_group_amount, to_group_amount, sparsity=None, is_sparse=False):
  """
  Creates a connection with uniform distribution.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  from_group_amount : int
      Number of cells in the 'from_group'.
  to_group_amount : int
      Number of cells in the 'to_group'.
  sparsity : float between 0 and 1
      Percentage of zeros in the matrix.
  is_sparse : Boolean
      Determines whether the returning Tensor is sparse or not.

  Returns
  -------
  out : Tensor
      Random adjacency matrix for TensorFlow.
  """
  connection_shape = (to_group_amount, from_group_amount)
  if is_sparse:
    indices = []
    values = []
    if sparsity is None:
      values = np.random.randint(2, size=connection_shape[0]*connection_shape[1]) * 2.0 - 1.0
      for i in range(connection_shape[0]):
        for j in range(connection_shape[1]):
          indices.append([i,j])
    else:
      for i in range(connection_shape[0]):
        for j in range(connection_shape[1]):
          if sparsity < np.random.random():
            indices.append([i,j])
            values.append(np.random.randint(2) * 2.0 - 1.0)

    initial = tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=[connection_shape[0], connection_shape[1]]), tf.float64)
  else:
    if sparsity is None:
      conn_matrix = np.random.randint(2, size=connection_shape) * 2.0 - 1.0
    else:
      conn_matrix = np.zeros(connection_shape)
      for i in range(connection_shape[0]):
        for j in range(connection_shape[1]):
          if sparsity < np.random.random():
            conn_matrix[i,j] = np.random.randint(2) * 2.0 - 1.0
    initial = conn_matrix

  return initial if is_sparse else tf.get_variable(name, initializer=initial)

def create_gaussian_connection(name, from_group_amount, to_group_amount, mean=0.0, std=1.0, sparsity=None, is_sparse=False):
  """
  Creates a connection with Gaussian distribution.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  from_group_amount : int
      Number of cells in the 'from_group'.
  to_group_amount : int
      Number of cells in the 'to_group'.
  mean : float
      Mean for the Gaussian distribution.
  std : float
      Standard deviation for the Gaussian distribution.
  sparsity : float between 0 and 1
      Percentage of zeros in the matrix.
  is_sparse : Boolean
      Determines whether the returning Tensor is sparse or not.

  Returns
  -------
  out : Tensor
      Random adjacency matrix for TensorFlow.
  """
  connection_shape = (to_group_amount, from_group_amount)
  if is_sparse:
    indices = []
    values = []
    if sparsity is None:
      values = np.random.normal(loc=mean, scale=std, size=connection_shape[0]*connection_shape[1])
      for i in range(connection_shape[0]):
        for j in range(connection_shape[1]):
          indices.append([i,j])
    else:
      for i in range(connection_shape[0]):
        for j in range(connection_shape[1]):
          if sparsity < np.random.random():
            indices.append([i,j])
            values.append(np.random.normal(loc=mean, scale=std))
    initial = tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=[connection_shape[0], connection_shape[1]]), tf.float64)
  else:
    if sparsity is None:
      conn_matrix = np.random.normal(loc=mean, scale=std, size=connection_shape)
    else:
      conn_matrix = np.zeros(connection_shape)
      for i in range(connection_shape[0]):
        for j in range(connection_shape[1]):
          if sparsity < np.random.random():
            conn_matrix[i,j] = np.random.normal(loc=mean, scale=std)
    initial = conn_matrix

  return initial if is_sparse else tf.get_variable(name, initializer=initial)
