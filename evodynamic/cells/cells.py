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

  def add_binary_state(self, state_name, init="random"):
#    assert init in ["random","central","zeros","ones","reversecentral"],\
#      "init must be 'random', 'central', 'reversecentral', 'zeros', or 'ones'."

    if init == "random":
      #np.random.seed(1)
      initial = np.random.randint(2, size=self.amount).astype(np.float64)
    elif init == "central":
      initial = np.zeros(self.amount).astype(np.float64)
      initial[int(self.amount//2)] = 1
    elif init == "zeros":
      initial = np.zeros(self.amount).astype(np.float64)
    elif init == "ones":
      initial = np.ones(self.amount).astype(np.float64)
    elif init == "reversecentral":
      initial = np.ones(self.amount).astype(np.float64)
      initial[int(self.amount//2)] = 0
    else:
      initial = init.reshape(-1).astype(np.float64)

    var = tf.get_variable(state_name, initializer=initial)
    self.states[state_name] = var
    return var

  def add_n_state(self, state_name, n_state):
    initial = np.random.randint(n_state, size=self.amount).astype(np.float64)
    var = tf.get_variable(state_name, initializer=initial)
    self.states[state_name] = var
    return var

  def add_real_state(self, state_name, stddev = .1):
    initial = tf.truncated_normal([self.amount], stddev=stddev,
                                  dtype=tf.dtypes.float64)
    var = tf.get_variable(state_name, initializer=initial)
    self.states[state_name] = var
    return var

  def get_shaped_indices(self):
    return np.arange(self.amount)#.reshape()
