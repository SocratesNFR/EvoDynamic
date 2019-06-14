""" Cells """

import tensorflow as tf
import numpy as np
from functools import reduce
from operator import mul
from typing import Tuple, Optional

class Cells(object):
  """
  Class for Cells
  """
  def __init__(self, amount: int, virtual_shape: Optional[Tuple[int]] = None):
    self.amount = amount
    self.virtual_shape = (self.amount,) if virtual_shape is None else virtual_shape
    assert self.amount == reduce(mul, self.virtual_shape),\
      "'amount' and 'virtual_shape' do not match"

    self.states = {}
    self.update_ops = []
    self.internal_connections = []
    self.external_state_name = ""

  def add_binary_state(self, state_name, init="random"):
    assert init == "random" or init == "central", "init must be 'random' or 'central'."

    if init == "random":
      initial = np.random.randint(2, size=self.amount).astype(np.float64)
    elif init == "central":
      initial = np.zeros(self.amount).astype(np.float64)
      initial[int(self.amount//2)] = 1

    var = tf.get_variable(state_name, initializer=initial)
    if len(self.states) == 0:
      self.external_state_state_name = state_name
    self.states[state_name] = var
    return var

  def add_n_state(self, state_name, n_state):
    initial = np.random.randint(n_state, size=self.amount).astype(np.float64)
    var = tf.get_variable(state_name, initializer=initial)
    if len(self.states) == 0:
      self.external_state_state_name = state_name
    self.states[state_name] = var
    return var

  def add_real_state(self, state_name, stddev = .1):
    initial = tf.truncated_normal([self.amount], stddev=stddev)
    var = tf.get_variable(state_name, initializer=initial)
    self.states[state_name] = var
    return var

  def set_external_state(self, state_name):
    state_name_exists = state_name in self.states
    if state_name_exists:
      self.external_state_name = state_name
    else:
      print("Warning: state_name for state does not exist.")
    return state_name_exists

  def get_shaped_indices(self):
    return np.arange(self.amount)#.reshape()

  def add_internal_connection(self,state_name,connection,activation_func=None,\
                              fargs=None):
    assert (state_name in self.states), "'state_name' must match existing state!"

    if fargs:
      _args = fargs
    else:
      _args = ()

    if isinstance(connection, tf.SparseTensor):
      res_matmul_op = tf.sparse.matmul(connection,\
                                       tf.transpose(tf.expand_dims(self.states[state_name],0)))
    else:
      res_matmul_op = tf.matmul(connection,\
                                tf.transpose(tf.expand_dims(self.states[state_name],0)))

    if activation_func == None:
      res_act_op = tf.assign(self.states[state_name], tf.squeeze(res_matmul_op))
    else:
      res_act_op = tf.assign(self.states[state_name],\
                             activation_func(tf.squeeze(res_matmul_op),\
                                             self.states[state_name], *_args))

    self.internal_connections.append(res_act_op)
