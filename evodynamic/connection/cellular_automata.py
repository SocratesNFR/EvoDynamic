""" Connections for cellular automata """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def create_count_neighbors_ca1d(width):
  """
  Returns a list with the weights for 'neighbors' and 'center_idx' parameters
  of evodynamic.connection.cellular_automata.create_conn_matrix_ca1d(...).
  The weights are responsible to count the number of alive neighbors.

  Parameters
  ----------
  width : int
      Neighborhood size.

  Returns
  -------
  out1 : list
      List of weights of the neighbors.
  out2 : int
      Index of the center of the neighborhood.
  """
  return [1 if p != width//2 else 0 for p in range(width)], width//2

def create_pattern_neighbors_ca1d(width, n_states=2):
  """
  Returns a list with the weights for 'neighbors' and 'center_idx' parameters
  of evodynamic.connection.cellular_automata.create_conn_matrix_ca1d(...).
  The weights are responsible to calculate an unique number for each different
  neighborhood pattern.

  Parameters
  ----------
  width : int
      Neighborhood size.
  n_states : int
      Number of discrete state in a cell.

  Returns
  -------
  out1 : list
      List of weights of the neighbors.
  out2 : int
      Index of the center of the neighborhood.
  """
  return [n_states**p for p in range(width)[::-1]], width//2

def create_conn_matrix_ca1d(name, width,\
                            neighbors=[4,2,1],\
                            center_idx=1, is_wrapped_ca = True,\
                            is_sparse = True):
  """
  This function creates a connection matrix for
  evodynamic.connection.WeightedConnection, so it can connect the cells as
  in a 1D cellular automaton.

  Parameters
  ----------
  name  : str
      Name of the Tensor.
  width : int
      Neighborhood size.
  neighbors  : list of int/float
      Weights for the neighbors.
  center_idx  : int
      Index of the center cell in the neighborhood.
  is_wrapped_ca  : Boolean
      Activates the wrapped boundary condition.
  is_sparse  :  Boolean
      Defines the type of Tensor variable for the connection matrix.

  Returns
  -------
  out : Tensor
      Connection matrix for TensorFlow.
  """

  neighbors_arr = np.array(neighbors)
  assert (len(neighbors_arr.shape) == 1),\
    "'neighbors' must be a list or numpy.ndarray with 1 dimensions!"

  nodes = width
  idx_dict_list = []
  for i in range(nodes):
    idx_dict_list.append({})
    for ii in range(-center_idx,neighbors_arr.shape[0]-center_idx):
      current_neighbor_cell = neighbors_arr[ii+center_idx]

      if (current_neighbor_cell != 0 and (is_wrapped_ca or \
      (not is_wrapped_ca and (0<=(i+ii)<width)))):
        idx_dict_list[-1][(i+ii)%width] = current_neighbor_cell

  if is_sparse:
    indices = []
    values = []
    for i, idx_dict in enumerate(idx_dict_list):
      for k in idx_dict:
        indices.append([i,k])
        values.append(idx_dict[k])
    initial = tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=[nodes, nodes]), tf.float64)
  else:
    conn_matrix = np.zeros((nodes, nodes))
    for i, idx_dict in enumerate(idx_dict_list):
      for k in idx_dict:
        conn_matrix[i,k] = idx_dict[k]
    initial = conn_matrix

  return initial if is_sparse else tf.get_variable(name, initializer=initial)

def create_count_neighbors_ca2d(width, height):
  """
  Returns a list with the weights for 'neighbors' and 'center_idx' parameters
  of evodynamic.connection.cellular_automata.create_conn_matrix_ca2d(...).
  The weights are responsible to count the number of alive neighbors.

  Parameters
  ----------
  width : int
      Neighborhood width.
  height : int
      Neighborhood height.

  Returns
  -------
  out1 : list
      List of weights of the neighbors.
  out2 : list
      Index of the center of the neighborhood.
  """
  center_idx_flat = width*(height//2) + width//2
  return np.array([1 if p != center_idx_flat else 0 for p in range(width*height)])\
        .reshape(width,height), [width//2, height//2]

def create_pattern_neighbors_ca2d(width, height, n_states=2):
  """
  Returns a list with the weights for 'neighbors' and 'center_idx' parameters
  of evodynamic.connection.cellular_automata.create_conn_matrix_ca1d(...).
  The weights are responsible to calculate an unique number for each different
  neighborhood pattern.

  Parameters
  ----------
  width : int
      Neighborhood width.
  height : int
      Neighborhood height.
  n_states : int
      Number of discrete state in a cell.

  Returns
  -------
  out1 : list
      List of weights of the neighbors.
  out2 : int
      Index of the center of the neighborhood.
  """
  return np.array([n_states**p for p in range(width*height)]).reshape(width,height),\
        [width//2, height//2]

def create_conn_matrix_ca2d(name, width, height,\
                            neighbors=[[0,1,0],[1,0,1],[0,1,0]],\
                            center_idx=[1,1], is_wrapped_ca = True,\
                            is_sparse = True):
  """
  This function creates a connection matrix for
  evodynamic.connection.WeightedConnection, so it can connect the cells as
  in a 2D cellular automaton.

  Parameters
  ----------
  name  : str
      Name of the Tensor.
  width : int
      Neighborhood width.
  height : int
      Neighborhood height.
  neighbors  : matrix or 2D list of int/float.
      Weights for the neighbors.
  center_idx  : list
      Index of the center cell in the neighborhood.
  is_wrapped_ca  : Boolean
      Activates the wrapped boundary condition.
  is_sparse  :  Boolean
      Defines the type of Tensor variable for the connection matrix.

  Returns
  -------
  out : Tensor
      Connection matrix for TensorFlow.
  """
  neighbors_arr = np.array(neighbors)
  assert (len(neighbors_arr.shape) == 2),\
    "'neighbors' must be a list or numpy.ndarray with 2 dimensions!"
  assert (len(center_idx) == 2),\
    "'center_idx' must be a list with 2 elements!"

  nodes = width* height
  idx_dict_list = []
  for i in range(nodes):
    idx_dict_list.append({})
    for ii in range(-center_idx[1],neighbors_arr.shape[0]-center_idx[1]):
      for jj in range(-center_idx[0],neighbors_arr.shape[1]-center_idx[0]):
        current_neighbor_cell = neighbors_arr[ii+center_idx[1], jj+center_idx[0]]

        if (current_neighbor_cell != 0 and (is_wrapped_ca or \
        (not is_wrapped_ca and (0 <= ((i%height)+ii) < width) and \
         (0 <= ((i//width)+jj) < height)))):
          idx_dict_list[-1][(i+ii)%width + (((i//width)+jj)%height)*width] =\
          current_neighbor_cell

  if is_sparse:
    indices = []
    values = []
    for i, idx_dict in enumerate(idx_dict_list):
      for k in idx_dict:
        indices.append([k,i])
        values.append(idx_dict[k])
    initial = tf.cast(tf.SparseTensor(indices=indices, values=values,\
                                      dense_shape=[nodes, nodes]), tf.float64)
  else:
    conn_matrix = np.zeros((nodes, nodes))
    for i, idx_dict in enumerate(idx_dict_list):
      for k in idx_dict:
        conn_matrix[k,i] = idx_dict[k]
    initial = conn_matrix

  return initial if is_sparse else tf.get_variable(name, initializer=initial)
