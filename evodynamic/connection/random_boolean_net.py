""" Connections for random Boolean network """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random

def create_pattern_neighbors(width, n_states=2):
  """
  This is a private function that returns the weights for calculating an unique
  number for each different neighborhood pattern in a random Boolean network.

  Parameters
  ----------
  width : int
      Neighborhood size.
  n_states : int
      Number of discrete state in a cell.

  Returns
  -------
  out : list
      List of weights of the neighbors.
  """
  return [n_states**p for p in range(width)[::-1]]

def create_conn_matrix(name, width, n_neighbors=3, n_states=2, is_sparse=True):
  """
  Creates a random square matrix with Gaussian distribution according to
  parameters for evodynamic.connection.WeightedConnection.

  Parameters
  ----------
  name : str
      Name of the Tensor.
  width : int
      Width of the adjacency matrix.
  n_neighbors : int
      Number of neighbors a cell will have.
  n_states : int
      Number of states a cell will have.
  is_sparse : Boolean
      Determines whether the returning Tensor is sparse or not.

  Returns
  -------
  out : Tensor
      Random adjacency matrix for TensorFlow.
  """
  pattern_neighbors = create_pattern_neighbors(n_neighbors, n_states=n_states)
  nodes = width
  if is_sparse:
    indices = []
    values = []
    for i in range(width):
      for idx, k in enumerate(random.sample(range(width), n_neighbors)):
        indices.append([i,k])
        values.append(pattern_neighbors[idx])
    initial = tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=[nodes, nodes]), tf.float64)
  else:
    conn_matrix = np.zeros((nodes, nodes))
    for i in range(width):
      for idx, k in enumerate(random.sample(range(width), n_neighbors)):
        conn_matrix[i,k] = pattern_neighbors[idx]
    initial = conn_matrix

  return initial if is_sparse else tf.get_variable(name, initializer=initial)
