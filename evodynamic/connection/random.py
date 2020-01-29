""" Connections for reservoir neural networks """

import tensorflow as tf
import numpy as np
from . import conn_utils

# Based on https://github.com/nschaetti/EchoTorch/blob/master/echotorch/nn/ESNCell.py

def create_gaussian_matrix(name, width, mean=0.0, std=1.0, sparsity=None, is_sparse=False):
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
    #print(indices)
    #print(values)
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
  return conn_utils.weight_variable_xavier_initialized([to_group_amount, from_group_amount], name=name)

def create_truncated_normal_connection(name, from_group_amount, to_group_amount, stddev=0.02):
  return conn_utils.weight_variable([to_group_amount, from_group_amount],
                                    stddev=stddev, name=name)
