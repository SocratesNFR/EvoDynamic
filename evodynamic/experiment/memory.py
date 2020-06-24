""" State memory """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class Memory(object):
  def __init__(self, experiment, state, memory_size) -> None:
    if memory_size < 2:
      raise Exception("Error: Memory size must be equal or larger than 1.")
    self.experiment = experiment
    self.state = state
    self.state_shape = self.state.shape
    self.memory_size = memory_size
    self.memory_shape = tuple([self.memory_size*self.state_shape[0]] +
                              list(np.array(self.state_shape)[1:]))
    self.state_memory = tf.get_variable(
          state.name.replace(":0", "")+"_memory",
          initializer=tf.zeros(self.memory_shape, dtype=tf.dtypes.float64))

  def get_op(self):
    return self.state_memory

  def get_state_memory(self):
    return self.experiment.session.run(self.state_memory)

  def reset(self):
    self.experiment.session.run(tf.assign(self.state_memory,
                                          tf.zeros_like(self.state_memory)))

  def update_state_memory(self):
    if self.experiment.memory_counter < self.memory_size:
#      update_indices = list(range(self.experiment.memory_counter*self.state_shape[0],
#                                  (self.experiment.memory_counter+1)*self.state_shape[0]))

      update_indices = [i for i in range(self.experiment.memory_counter*self.state_shape[0],\
                        (self.experiment.memory_counter+1)*self.state_shape[0])]

      self.experiment.session.run(
          tf.scatter_update(self.state_memory, update_indices, self.state))

    else:
      selection_indices = [[i] for i in range(self.state_shape[0],\
                           self.memory_size*self.state_shape[0])]

      memory_shift_selection_op = tf.gather_nd(self.state_memory,selection_indices)

      state_memory_update_indices = [i for i in range(self.memory_size*self.state_shape[0])]
      concat_op = tf.concat([memory_shift_selection_op, self.state], 0)

      state_memory_update_op = tf.scatter_update(self.state_memory,
                                                 state_memory_update_indices,
                                                 concat_op)
      self.experiment.session.run(state_memory_update_op)

