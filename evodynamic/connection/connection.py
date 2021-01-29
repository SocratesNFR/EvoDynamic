""" Connection """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class BaseConnection(object):
  def __init__(self, from_group_state, to_group_state, activation_func):
    self.from_group_state = from_group_state
    self.to_group_state = to_group_state
    self.activation_func = activation_func
    self.experiment = None
    self.list_ops = []
    self.assign_output = None
    self.output = None
    self.is_input = from_group_state.op.type == "Placeholder"

  def compute(self):
    raise NotImplementedError

  def set_experiment(self, experiment):
    raise NotImplementedError

  def __get_ops(self):
    raise NotImplementedError

  def __get_output(self):
    raise NotImplementedError

class IndexConnection(BaseConnection):
  def __init__(self, from_group_state, to_group_state, to_group_state_idx, activation_func=tf.scatter_update):
    super().__init__(from_group_state, to_group_state, activation_func)
    self.to_group_state_idx = tf.convert_to_tensor(to_group_state_idx, tf.int64)

  def set_experiment(self, experiment, is_first_training_connection=False):
    self.experiment = experiment
    if not is_first_training_connection:
      for exp_conn in self.experiment.connection_list:
        if exp_conn.to_group_state == self.from_group_state:
          self.from_group_state = exp_conn.assign_output
    self.list_ops = self.__get_ops()[0]
    self.assign_output = self.__get_output()[0]
    self.output = self.__get_output()[1]

  def __get_ops(self):
    if self.is_input:
      ops = [tf.cond(self.experiment.has_input,
                     true_fn=lambda: self.activation_func(self.to_group_state, self.to_group_state_idx, self.from_group_state),
                     false_fn=lambda: self.to_group_state)]
    else:
      ops = [self.activation_func(self.to_group_state, self.to_group_state_idx, self.from_group_state)]
    return ops, ops

  def __get_output(self):
    if self.is_input:
      output = tf.cond(self.experiment.has_input,
                       true_fn=lambda: self.activation_func(self.to_group_state, self.to_group_state_idx, self.from_group_state),
                       false_fn=lambda: self.to_group_state)
    else:
      output = self.activation_func(self.to_group_state, self.to_group_state_idx, self.from_group_state)
    return output, output

class GatherIndexConnection(BaseConnection):
  def __init__(self, from_group_state, to_group_state, to_group_state_idx):

    def tf_gather_nd_update(to_group_state, to_group_state_idx, from_group_state):
      g_op = tf.gather(from_group_state, to_group_state_idx)
      return tf.assign(to_group_state,g_op)

    super().__init__(from_group_state, to_group_state, tf_gather_nd_update)
    self.to_group_state_idx = tf.convert_to_tensor(to_group_state_idx, tf.int64) # Tensor of type int32 or int64

  def set_experiment(self, experiment, is_first_training_connection=False):
    self.experiment = experiment
    if not is_first_training_connection:
      for exp_conn in self.experiment.connection_list:
        if exp_conn.to_group_state == self.from_group_state:
          self.from_group_state = exp_conn.assign_output
    self.list_ops = self.__get_ops()[0]
    self.assign_output = self.__get_output()[0]
    self.output = self.__get_output()[1]

  def __get_ops(self):
    if self.is_input:
      ops = [tf.cond(self.experiment.has_input,
                     true_fn=lambda: self.activation_func(self.to_group_state, self.to_group_state_idx, self.from_group_state),
                     false_fn=lambda: self.to_group_state)]
    else:
      ops = [self.activation_func(self.to_group_state, self.to_group_state_idx, self.from_group_state)]
    return ops, ops

  def __get_output(self):
    if self.is_input:
      output = tf.cond(self.experiment.has_input,
                       true_fn=lambda: self.activation_func(self.to_group_state, self.to_group_state_idx, self.from_group_state),
                       false_fn=lambda: self.to_group_state)
    else:
      output = self.activation_func(self.to_group_state, self.to_group_state_idx, self.from_group_state)
    return output, output

class WeightedConnection(BaseConnection):
  def __init__(self, from_group_state, to_group_state, activation_func, w, fargs_list=None):
    # fargs_list: List of Tuples
    super().__init__(from_group_state, to_group_state, activation_func)
    self.w = w
    self.fargs_list = fargs_list if fargs_list else [()]

  def set_experiment(self, experiment, is_first_training_connection=False):
    self.experiment = experiment
    if not is_first_training_connection:
      for exp_conn in self.experiment.connection_list:
        if exp_conn.to_group_state == self.from_group_state:
          self.from_group_state = exp_conn.assign_output
    self.list_ops = self.__get_ops()[0]
    self.assign_output = self.__get_output()[0]
    self.output = self.__get_output()[1]

  def compute(self):
    for fargs in self.fargs_list:
      if isinstance(self.w, tf.SparseTensor):
        res_matmul_op = tf.sparse.matmul(self.w, self.from_group_state)
      else:
        res_matmul_op = tf.matmul(self.w, self.from_group_state)

      if self.activation_func == None:
        res_act_op = res_matmul_op
      else:
        res_act_op = self.activation_func(res_matmul_op, self.from_group_state, *fargs)

      res_assign_op = tf.assign(self.to_group_state, res_act_op)
      self.experiment.session.run(res_assign_op)

  def __get_ops(self):
    list_ops = []
    output_ops = []
    for fargs in self.fargs_list:
      if isinstance(self.w, tf.SparseTensor):
        res_matmul_op = tf.sparse.matmul(self.w, self.from_group_state)
      else:
        res_matmul_op = tf.matmul(self.w, self.from_group_state)

      if self.activation_func == None:
        res_act_op = res_matmul_op
      else:
        res_act_op = self.activation_func(res_matmul_op,\
                                          self.from_group_state, *fargs)

      res_assign_op = tf.assign(self.to_group_state, res_act_op)

      list_ops.append(res_assign_op)
      output_ops.append(res_act_op)
    return list_ops, output_ops

  def __get_output(self):
    list_ops = []
    output_ops = []
    for fargs in self.fargs_list:
      from_op = self.from_group_state if len(list_ops) == 0 else list_ops[-1]
      if isinstance(self.w, tf.SparseTensor):
        res_matmul_op = tf.sparse.matmul(self.w, from_op)
      else:
        res_matmul_op = tf.matmul(self.w, from_op)

      if self.activation_func == None:
        res_act_op = res_matmul_op
      else:
        res_act_op = self.activation_func(res_matmul_op,\
                                          self.from_group_state, *fargs)

      res_assign_op = tf.assign(self.to_group_state, res_act_op)

      list_ops.append(res_assign_op)
      output_ops.append(res_act_op)
    return list_ops[-1], output_ops[-1]