""" Cells """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from functools import reduce
from operator import mul
from typing import Tuple, Optional

class Cells(object):
  """
  Class for Cells
  """
  def __init__(self, amount: int, batch_size: int, virtual_shape: Optional[Tuple[int]] = None):
    self.amount = amount
    self.batch_size = batch_size
    self.amount_with_batch = (self.amount, self.batch_size)
    self.virtual_shape = (self.amount,) if virtual_shape is None else virtual_shape
    assert self.amount == reduce(mul, self.virtual_shape),\
      "'amount' and 'virtual_shape' do not match"

    self.states = {}
    self.update_ops = []

  def add_binary_state(self, state_name, init="random"):
#    assert init in ["random","central","zeros","ones","reversecentral"],\
#      "init must be 'random', 'central', 'reversecentral', 'zeros', or 'ones'."

    if init == "random":
      #np.random.seed(1)
      initial = np.random.randint(2, size=self.amount_with_batch).astype(np.float64)
    elif init == "central":
      initial = np.zeros(self.amount_with_batch).astype(np.float64)
      initial[int(self.amount//2),:] = 1
    elif init == "zeros":
      initial = np.zeros(self.amount_with_batch).astype(np.float64)
    elif init == "ones":
      initial = np.ones(self.amount_with_batch).astype(np.float64)
    elif init == "reversecentral":
      initial = np.ones(self.amount_with_batch).astype(np.float64)
      initial[int(self.amount//2),:] = 0
    else: # Manual initialization with list or numpy array
      init = np.array(init)
      virtual_shape_list = list(self.virtual_shape)
      virtual_shape_list.insert(1,self.batch_size)
      if init.shape == (self.amount,):
        initial = np.hstack(self.batch_size*[init]).astype(np.float64)
      elif init.shape == self.virtual_shape:
        initial = np.hstack(self.batch_size*[init.reshape(-1)]).astype(np.float64)
      elif init.shape == self.amount_with_batch:
        initial = init.astype(np.float64)
      elif init.shape == tuple(virtual_shape_list):
        initial = init.reshape(-1,self.batch_size).astype(np.float64)

    var = tf.get_variable(state_name, initializer=initial)
    self.states[state_name] = var
    return var

  def add_n_state(self, state_name, n_state):
    initial = np.random.randint(n_state, size=self.amount_with_batch).astype(np.float64)
    var = tf.get_variable(state_name, initializer=initial)
    self.states[state_name] = var
    return var

  def add_real_state(self, state_name, stddev = .1):
    initial = tf.truncated_normal(self.amount_with_batch, stddev=stddev,
                                  dtype=tf.dtypes.float64)
    var = tf.get_variable(state_name, initializer=initial)
    self.states[state_name] = var
    return var

  def get_shaped_indices(self):
    return np.arange(self.amount)#.reshape()
