""" Connection """

import tensorflow as tf
#import numpy as np

class BaseConnection(object):
  def __init__(self, from_group, to_group, activation_func):
    self.from_group = from_group
    self.to_group = to_group
    self.activation_func = activation_func
    self.experiment = None

  def compute(self):
    raise NotImplementedError

  def set_experiment(self, experiment):
    self.experiment = experiment

#TODO: function tf.scatter_update or my_var = my_var[4:8].assign(tf.zeros(4))
class IndexConnection(BaseConnection):
  def __init__(self, from_group, to_group, to_group_idx, activation_func=tf.scatter_update):
    BaseConnection.__init__(self, from_group, to_group, activation_func)
    self.to_group_idx = to_group_idx # Tensor of type int32 or int64

  def compute(self):
    raise NotImplementedError

#TODO: Rest of class WeightedConnection
class WeightedConnection(BaseConnection):
  def __init__(self, from_group, to_group, activation_func, w, fargs_list=None):
    # fargs_list: List of Tuples
    BaseConnection.__init__(self, from_group, to_group, activation_func)
    self.w = w
    self.fargs_list = fargs_list if fargs_list else [()]
    #if self.w is None:

  def compute(self):
    for fargs in self.fargs_list:
      if isinstance(self.w, tf.SparseTensor):
        res_matmul_op = tf.sparse.matmul(self.w,\
                                         tf.transpose(tf.expand_dims(self.from_group,0)))
      else:
        res_matmul_op = tf.matmul(self.w,\
                                  tf.transpose(tf.expand_dims(self.from_group,0)))
  
      if self.activation_func == None:
        res_act_op = tf.assign(self.to_group, tf.squeeze(res_matmul_op))
      else:
        res_act_op = tf.assign(self.to_group,\
                               self.activation_func(tf.squeeze(res_matmul_op),\
                                               self.from_group, *fargs))
      self.experiment.session.run(res_act_op)
