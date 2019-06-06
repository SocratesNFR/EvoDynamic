""" Experiment """

import tensorflow as tf
import evodynamic.cells as cells
import evodynamic.utils as utils
from . import monitor


class Experiment(object):
  def __init__(self, dt: float = 1.0) -> None:
    tf.reset_default_graph()
    self.dt = dt
    self.cell_groups = {}
    self.connections = {}
    self.monitors = {}
    self.session = tf.Session()

  def add_group_cells(self,name, amount):
    g_cells = cells.Cells(amount)
    self.cell_groups[name] = g_cells
    return g_cells

  def initialize_cells(self):
    self.session.run(tf.global_variables_initializer())

    for monitor_key in self.monitors:
      self.monitors[monitor_key].initialize()

  def close(self):
    self.session.close()

  def run(self,timesteps: int = 10):
    for step in range(timesteps-1):
      for group_key in self.cell_groups:
        self.session.run(self.cell_groups[group_key].internal_connections)

      for monitor_key in self.monitors:
        self.monitors[monitor_key].record()

      utils.progressbar(step+1, timesteps-1)

  def run_step(self):
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