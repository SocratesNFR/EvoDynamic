""" Experiment """

import tensorflow as tf
from . import monitor
from . import memory
from .. import cells
from .. import utils

class Experiment(object):
  def __init__(self, dt: float = 1.0, input_delay=0, training_delay=0) -> None:
    tf.reset_default_graph()
    self.dt = dt
    self.cell_groups = {}
    self.connections = {}
    self.trainable_connections = {}
    self.connection_ops = []
    self.input_name_list = []
    self.connection_ops_no_inputs = []
    self.train_ops = []
    self.monitors = {}
    self.session = tf.Session()
    self.memories = {}
    self.step_counter = 0
    self.input_delay = input_delay
    self.training_delay = training_delay

  def add_input(self, dtype, shape, name):
    input_placeholder = tf.placeholder(dtype, shape=shape, name=name)
    self.input_name_list.append(name)
    return input_placeholder

  def add_group_cells(self, name, amount):
    g_cells = cells.Cells(amount)
    self.cell_groups[name] = g_cells
    return g_cells

  def add_cells(self, name, g_cells):
    self.cell_groups[name] = g_cells
    return g_cells

  def add_state_memory(self, state, memory_size):
    state_memory = memory.Memory(self,state,memory_size)
    self.memories[state] = state_memory
    return state_memory.get_op()

  def add_connection(self, name, connection):
    connection.set_experiment(self)
    self.connections[name] = connection
    self.connection_ops.append(connection.list_ops)
    if not connection.from_group.name.split(":")[0] in self.input_name_list: # if not input
      self.connection_ops_no_inputs.append(connection.list_ops)
    return connection

  def add_trainable_connection(self, name, connection):
    self.add_connection(name, connection)
    self.trainable_connections[name] = connection
    return connection

  def initialize_cells(self):
    self.session.run(tf.global_variables_initializer())
    for monitor_key in self.monitors:
      self.monitors[monitor_key].initialize()

  def set_training(self, loss, learning_rate, optimizer="adam"):
    model_vars = tf.trainable_variables()
    t_vars = []
    for var in model_vars:
      for conn_key in self.trainable_connections:
        if conn_key in var.name:
          t_vars.append(var)

    if optimizer == "adam":
      train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=t_vars)
    else:
      print("set_training has set invalid optimizer")

    self.train_ops.append(train_op)

  def close(self):
    self.session.close()

  def run(self,timesteps: int = 10, feed_dict=None):
    for step in range(timesteps-1):
      self.run_step(feed_dict=feed_dict)
      utils.progressbar(step+1, timesteps-1)

  def run_step(self, feed_dict=None):
    self.step_counter += 1
    if self.step_counter % (self.input_delay+1) == 0:
      for conn_key in self.connections:
        self.session.run(self.connection_ops, feed_dict=feed_dict)
    else:
      for conn_key in self.connections:
        self.session.run(self.connection_ops_no_inputs, feed_dict=feed_dict)

    for memory_key in self.memories:
      self.memories[memory_key].update_state_memory()

    for monitor_key in self.monitors:
      self.monitors[monitor_key].record()

    if self.step_counter % (self.training_delay+1) == self.training_delay:
      self.session.run(self.train_ops, feed_dict=feed_dict)

  def check_group_cells_state(self, group_cells_name, state_name):
    group_cells_name_exists = group_cells_name in self.cell_groups
    assert group_cells_name_exists, "Error: group_cells_name for group_cells does not exist."

    state_name_exists = state_name in self.cell_groups[group_cells_name].states
    assert state_name_exists, "Error: state_name for state does not exist."

  def get_group_cells_state(self, group_cells_name, state_name):
    self.check_group_cells_state(group_cells_name, state_name)

    return self.session.run(self.cell_groups[group_cells_name].states[state_name])

  def add_monitor(self, group_cells_name, state_name, timesteps=None):
    self.check_group_cells_state(group_cells_name, state_name)

    self.monitors[(group_cells_name,state_name)] =\
      monitor.Monitor(self, group_cells_name, state_name, duration=timesteps)

  def get_monitor(self, group_cells_name, state_name):
    self.check_group_cells_state(group_cells_name, state_name)

    return self.monitors[(group_cells_name,state_name)].get()

  def get_connection(self, conn_name):
    conn_name_exists = conn_name in self.connections
    assert conn_name_exists, "Error: conn_name for connections does not exist."

    return self.connections[conn_name]