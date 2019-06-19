""" Experiment """

import tensorflow as tf
from . import monitor
from .. import cells
from .. import utils


class Experiment(object):
  def __init__(self, dt: float = 1.0) -> None:
    tf.reset_default_graph()
    self.dt = dt
    self.cell_groups = {}
    self.connections = {}
    self.connection_ops = []
    self.monitors = {}
    self.session = tf.Session()

  def add_input(self, dtype, shape, name):
    input_placeholder = tf.placeholder(dtype, shape=shape, name=name)
    return input_placeholder

  def add_group_cells(self, name, amount):
    g_cells = cells.Cells(amount)
    self.cell_groups[name] = g_cells
    return g_cells

  def add_cells(self, name, g_cells):
    self.cell_groups[name] = g_cells
    return g_cells

  def add_connection(self, name, connection):
    connection.set_experiment(self)
    self.connections[name] = connection
    self.connection_ops.append(connection.list_ops)
    return connection

  def initialize_cells(self):
    self.session.run(tf.global_variables_initializer())

    for monitor_key in self.monitors:
      self.monitors[monitor_key].initialize()

  def close(self):
    self.session.close()

  #TODO: def run(self,timesteps: int = 10, dataset, batch_size):
  def run(self,timesteps: int = 10, feed_dict=None):
    for step in range(timesteps-1):
      self.session.run(self.connection_ops, feed_dict=feed_dict)

      for group_key in self.cell_groups:
        self.session.run(self.cell_groups[group_key].internal_connections)

      for monitor_key in self.monitors:
        self.monitors[monitor_key].record()

      utils.progressbar(step+1, timesteps-1)

  #TODO: def run_step(self, input_dict):
  def run_step(self, feed_dict=None):
    for conn_key in self.connections:
      self.session.run(self.connection_ops, feed_dict=feed_dict)

    for group_key in self.cell_groups:
      self.session.run(self.cell_groups[group_key].internal_connections)

    for monitor_key in self.monitors:
      self.monitors[monitor_key].record()

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