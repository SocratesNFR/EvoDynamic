""" Connection """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class BaseConnection(object):
  def __init__(self, from_group_state, to_group_state, activation_func):
    """
    BaseConnection constructor

    Parameters
    ----------
    from_group_state : Tensor
        State in group of cells that will affect 'to_group_state'.
    to_group_state : Tensor
        State in group of cells that will be affected by 'from_group_state'.
    activation_func : function
        Function that defines how 'to_group_state' will be affected.

    Returns
    -------
    out : object
        New object of class BaseConnection.
    """
    self.from_group_state = from_group_state
    self.to_group_state = to_group_state
    self.activation_func = activation_func
    self.experiment = None
    self.list_ops = []
    self.assign_output = None
    self.output = None
    self.is_input = from_group_state.op.type == "Placeholder"

  def set_experiment(self, experiment, is_first_training_connection):
    """
    Used by evodynamic.experiment.experiment to give the Connection object the
    Experiment object that it was added.
    """
    raise NotImplementedError

  def __get_ops(self):
    """
    Returns a tuple with 2 lists of TensorFlow operations used to execute
    the connection. If the last operation is tf.assign, then the first one has
    the tf.assign, and the other the Tensor given to the tf.assign. If there is
    no tf.assign, the 2 lists are equal.
    """
    raise NotImplementedError

  def __get_output(self):
    """
    Returns a tuple with 2 TensorFlow operations or Tensors. They are the last
    operation or Tensor of the '__get_ops' function. If the last operation is
    tf.assign, then the first one has the tf.assign, and the other the Tensor
    given to the tf.assign. If there is no tf.assign, the 2 lists are equal.
    """
    raise NotImplementedError

class IndexConnection(BaseConnection):
  def __init__(self, from_group_state, to_group_state, to_group_state_idx):
    """
    IndexConnection constructor

    Parameters
    ----------
    from_group_state : Tensor
        State in group of cells that will affect 'to_group_state'.
    to_group_state : Tensor
        State in group of cells that will be affected by 'from_group_state'.
    to_group_state_idx : sequence of ints
        Sequence of indices in 'to_group_state' where the values of
        'from_group_state' will be assigned in 'to_group_state'.

    Returns
    -------
    out : object
        New object of class IndexConnection.
    """
    super().__init__(from_group_state, to_group_state, tf.scatter_update)
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
    list_ops, output_ops = self.__get_ops()
    return list_ops[-1], output_ops[-1]

class GatherIndexConnection(BaseConnection):
  def __init__(self, from_group_state, to_group_state, from_group_state_idx):
    """
    IndexConnection constructor

    Parameters
    ----------
    from_group_state : Tensor
        State in group of cells that will affect 'to_group_state'.
    to_group_state : Tensor
        State in group of cells that will be affected by 'from_group_state'.
    from_group_state_idx : sequence of ints
        Sequence of indices in 'from_group_state' where the values of
        'from_group_state' will be assigned in 'to_group_state'.

    Returns
    -------
    out : object
        New object of class IndexConnection.
    """

    def tf_gather_nd_update(to_group_state, from_group_state_idx, from_group_state):
      g_op = tf.gather(from_group_state, from_group_state_idx)
      return tf.assign(to_group_state,g_op)

    super().__init__(from_group_state, to_group_state, tf_gather_nd_update)
    self.from_group_state_idx = tf.convert_to_tensor(from_group_state_idx, tf.int64) # Tensor of type int32 or int64

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
                     true_fn=lambda: self.activation_func(self.to_group_state, self.from_group_state_idx, self.from_group_state),
                     false_fn=lambda: self.to_group_state)]
    else:
      ops = [self.activation_func(self.to_group_state, self.from_group_state_idx, self.from_group_state)]
    return ops, ops

  def __get_output(self):
    list_ops, output_ops = self.__get_ops()
    return list_ops[-1], output_ops[-1]

class WeightedConnection(BaseConnection):
  def __init__(self, from_group_state, to_group_state, activation_func, w, fargs_list=None):
    """
    WeightedConnection constructor

    Parameters
    ----------
    from_group_state : Tensor
        State in group of cells that will affect 'to_group_state'.
    to_group_state : Tensor
        State in group of cells that will be affected by 'from_group_state'.
    activation_func : function
        Function that defines how 'to_group_state' will be affected.
    w : Tensor
        Weighted adjacency matrix.
    fargs_list: list, optional
        List of arguments to be added to some 'activation_func'.

    Returns
    -------
    out : object
        New object of class WeightedConnection.

    Examples
    --------
    >>> WeightedConnection(ca_binary_state,ca_binary_state,
                           rule_binary_ca_1d_width3_func,
                           ca_conn,
                           fargs_list=[(rule,)])
    """
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
    list_ops, output_ops = self.__get_ops()
    return list_ops[-1], output_ops[-1]
