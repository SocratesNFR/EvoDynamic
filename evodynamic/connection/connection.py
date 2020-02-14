""" Connection """

import tensorflow as tf

class BaseConnection(object):
  def __init__(self, from_group, to_group, activation_func):
    self.from_group = from_group
    self.to_group = to_group
    self.activation_func = activation_func
    self.experiment = None
    self.list_ops = []
    self.output = None

  def compute(self):
    raise NotImplementedError

  def set_experiment(self, experiment):
    self.experiment = experiment

  def __get_ops(self):
    raise NotImplementedError

class IndexConnection(BaseConnection):
  def __init__(self, from_group, to_group, to_group_idx, activation_func=tf.scatter_update):
    super().__init__(from_group, to_group, activation_func)
    self.to_group_idx = tf.convert_to_tensor(to_group_idx, tf.int64) # Tensor of type int32 or int64
    self.list_ops = self.__get_ops()

  def __get_ops(self):
    return [self.activation_func(self.to_group, self.to_group_idx, self.from_group)]

class GatherIndexConnection(BaseConnection):
  def __init__(self, from_group, to_group, to_group_idx):

    def tf_gather_nd_update(to_group, to_group_idx, from_group):
      g_op = tf.gather(from_group, to_group_idx)
      return tf.assign(to_group,g_op)

    super().__init__(from_group, to_group, tf_gather_nd_update)
    self.to_group_idx = tf.convert_to_tensor(to_group_idx, tf.int64) # Tensor of type int32 or int64
    self.list_ops = self.__get_ops()

  def __get_ops(self):
    return [self.activation_func(self.to_group, self.to_group_idx, self.from_group)]


class WeightedConnection(BaseConnection):
  def __init__(self, from_group, to_group, activation_func, w, fargs_list=None):
    # fargs_list: List of Tuples
    super().__init__(from_group, to_group, activation_func)
    self.w = w
    self.fargs_list = fargs_list if fargs_list else [()]
    self.list_ops = self.__get_ops()[0]
    self.output = self.__get_ops()[1][-1]

  def compute(self):
    for fargs in self.fargs_list:
      if isinstance(self.w, tf.SparseTensor):
        res_matmul_op = tf.sparse.matmul(self.w,\
                                         tf.transpose(tf.expand_dims(self.from_group,0)))
      else:
        res_matmul_op = tf.matmul(self.w,\
                                  tf.transpose(tf.expand_dims(self.from_group,0)))

      if self.activation_func == None:
        res_act_op = tf.squeeze(res_matmul_op)
      else:
        res_act_op = self.activation_func(tf.squeeze(res_matmul_op),
                                          self.from_group, *fargs)

      res_assign_op = tf.assign(self.to_group, tf.squeeze(res_act_op))
      self.experiment.session.run(res_assign_op)

  def __get_ops(self):
    list_ops = []
    output_ops = []
    for fargs in self.fargs_list:
      if isinstance(self.w, tf.SparseTensor):
        res_matmul_op = tf.sparse.matmul(self.w,\
                                         tf.transpose(tf.expand_dims(self.from_group,0)))
      else:
        res_matmul_op = tf.matmul(self.w,\
                                  tf.transpose(tf.expand_dims(self.from_group,0)))

      if self.activation_func == None:
        res_act_op = tf.squeeze(res_matmul_op)
      else:
        res_act_op = self.activation_func(tf.squeeze(res_matmul_op),\
                                          self.from_group, *fargs)

      res_assign_op = tf.assign(self.to_group, tf.squeeze(res_act_op))

      list_ops.append(res_assign_op)
      output_ops.append(res_act_op)
    return list_ops, output_ops