""" State memory """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Memory(object):
  def __init__(self, experiment, state, memory_size) -> None:
    """
    Memory constructor.

    A memory saves some time steps of a state for being used by other group of
    cells.

    Parameters
    ----------
    experiment : object
        Experiment object.
    state : Tensor
        Tensor of the state to be in memory.
    memory_size : int
        Number of time steps to be saved in memory.

    Returns
    -------
    out : object
        New object of class Memory.
    """
    if memory_size < 2:
      raise Exception("Error: Memory size must be equal or larger than 1.")

    self.experiment = experiment
    self.state_shape = state.get_shape().as_list()
    self.memory_size = memory_size
    self.memory_shape = tuple([self.memory_size*self.state_shape[0]] +
                              list(np.array(self.state_shape)[1:]))
    self.state_memory = tf.get_variable(
          state.name.replace(":0", "")+"_memory",
          initializer=tf.zeros(self.memory_shape, dtype=tf.dtypes.float64))

    self.selection_indices = [[i] for i in range(self.state_shape[0],\
                           self.memory_size*self.state_shape[0])]
    self.from_group_state = state
    for exp_conn in self.experiment.connection_list:
      if exp_conn.to_group_state == self.from_group_state:
        self.from_group_state = exp_conn.assign_output
    self.to_group_state = self.get_op()
    self.assign_output = self.get_op()

  def get_op(self):
    """
    Get the Tensor operation for the memory and its update.
    Update the state memory in the current time step. If memory is full, remove
    the oldest data and stack the new one.

    Returns
    -------
    out : Tensor
        Tensor operation for the memory and its update.
    """

    memory_shift_selection_op = tf.gather_nd(self.state_memory,self.selection_indices)
    concat_op = tf.concat([memory_shift_selection_op, self.from_group_state], 0)
    state_memory_update_op = tf.assign(self.state_memory, concat_op)

    return state_memory_update_op

  def get_state_memory(self):
    """
    Get the value stored in the memory.

    Returns
    -------
    out : Numpy array
        Array with the value of the memory.
    """
    return self.experiment.session.run(self.state_memory)

  def reset(self):
    """
    Reset the memory.
    """
    self.experiment.session.run(self.state_memory.initializer)

  def reset_op(self):
    """
    Returns the TensorFlow operation for resetting the Tensor of the memory.
    """
    return [self.state_memory.initializer]
