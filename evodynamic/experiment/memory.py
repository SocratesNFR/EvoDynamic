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
    self.state_shape = self.state.get_shape().as_list()
    print(" self.state_shape", self.state_shape)
    self.memory_size = memory_size
    self.memory_shape = tuple([self.memory_size*self.state_shape[0]] +
                              list(np.array(self.state_shape)[1:]))
    self.state_memory = tf.get_variable(
          state.name.replace(":0", "")+"_memory",
          initializer=tf.zeros(self.memory_shape, dtype=tf.dtypes.float64))

    self.selection_indices = [i for i in range(self.state_shape[0],\
                           self.memory_size*self.state_shape[0])]

#    self.total_update_indices = [[i] for i in range(self.memory_size*self.state_shape[0])]

    self.from_group = state
    self.to_group = self.state_memory

    self.assign_output = self.update_state_memory_act()
    for exp_conn in self.experiment.connection_list:
      if exp_conn.to_group == self.from_group:
        self.from_group = exp_conn.assign_output

  def get_op(self):
    return self.state_memory

  def get_state_memory(self):
    return self.experiment.session.run(self.state_memory)

  def reset(self):
    self.counter = 0
    self.experiment.session.run(tf.assign(self.state_memory,
                                          tf.zeros_like(self.state_memory)))

  def reset_op(self):
    self.counter = 0
    return [tf.assign(self.state_memory, tf.zeros_like(self.state_memory))]


#  def update_state_memory(self):
#    if self.counter < self.memory_size:
#      update_indices = [i for i in range(self.counter*self.state_shape[0],\
#                        (self.counter+1)*self.state_shape[0])]
#
#      self.experiment.session.run(
#          tf.scatter_update(self.state_memory, update_indices, self.state))
#    else:
#      memory_shift_selection_op = tf.gather_nd(self.state_memory,self.selection_indices)
#
#      concat_op = tf.concat([memory_shift_selection_op, self.state], 0)
#
#      state_memory_update_op = tf.scatter_update(self.state_memory,
#                                                 self.total_update_indices,
#                                                 concat_op)
#      self.experiment.session.run(state_memory_update_op)
#    self.counter +=1
#
#  def update_state_memory_op(self):
#    if self.counter < self.memory_size:
#      update_indices = [i for i in range(self.counter*self.state_shape[0],\
#                        (self.counter+1)*self.state_shape[0])]
#
#      state_memory_update_op = tf.scatter_update(self.state_memory, update_indices, self.state)
#    else:
#      memory_shift_selection_op = tf.gather_nd(self.state_memory,self.selection_indices)
#
#      concat_op = tf.concat([memory_shift_selection_op, self.state], 0)
#
#      state_memory_update_op = tf.scatter_update(self.state_memory,
#                                                 self.total_update_indices,
#                                                 concat_op)
#    self.counter +=1
#    return [state_memory_update_op]


  def update_state_memory_act(self):
    # if self.counter < self.memory_size
#    update_indices = tf.range(tf.multiply(self.counter_tf, self.state_shape[0]),
#                              tf.multiply(tf.add(self.counter_tf,1), self.state_shape[0]),
#                              dtype=tf.int64)
#
#    state_memory_update_not_full_op = tf.scatter_update(self.state_memory, update_indices, self.state)
    # end if self.counter < self.memory_size

    # if self.counter >= self.memory_size
    memory_shift_selection_op = tf.gather(self.state_memory,self.selection_indices)

    concat_op = tf.concat([memory_shift_selection_op, self.state], 0)

#    state_memory_update_full_op = tf.scatter_update(self.state_memory,
#                                                 self.total_update_indices,
#                                                 concat_op)


#    state_memory_update_full_op = tf.tensor_scatter_nd_update(self.state_memory,
#                                                     self.total_update_indices,
#                                                     concat_op)


    state_memory_update_full_op = tf.assign(self.state_memory, concat_op)



    return state_memory_update_full_op
    # end if self.counter >= self.memory_size

#    memory_update_cond_op = tf.less_equal(tf.assign_add(self.counter_tf,1), self.memory_size)
#    state_memory_update_op = tf.cond(memory_update_cond_op,
#                                     lambda: state_memory_update_not_full_op,
#                                     lambda: state_memory_update_full_op)

#    return state_memory_update_op
