""" State memory """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Memory(object):
  def __init__(self, experiment, state, memory_size) -> None:
    self.experiment = experiment
    self.state = state
    self.memory_size = memory_size
    self.state_memory = []
    for i in range(memory_size):
      self.state_memory.append(tf.get_variable(
          state.name.replace(":0", "")+"_"+str(i),
          initializer=tf.zeros(state.shape, dtype=tf.dtypes.float64)))

    self.memory_op = tf.concat(self.state_memory,0)

  def get_op(self):
    return self.memory_op

  def get_state_memory(self):
    return self.experiment.session.run(self.memory_op)

  def update_state_memory(self):
    if self.experiment.step_counter < len(self.state_memory):
      self.experiment.session.run(
          tf.assign(self.state_memory[self.experiment.step_counter], self.state))
    else:
      state_memory_part = self.state_memory.pop(0)
      self.experiment.session.run(tf.assign(state_memory_part, self.state))
      self.state_memory.append(state_memory_part)

