""" Experiment """

import tensorflow as tf
import numpy as np
import evodynamic.cells as cells
import evodynamic.utils as utils

class Experiment(object):
  def __init__(self, dt: float = 1.0) -> None:
    tf.reset_default_graph()
    self.dt = dt
    self.cell_groups = {}
    self.connections = {}
    self.monitors = {}
    self.session = tf.InteractiveSession()

  def add_group_cells(self,name, amount):
    g_cells = cells.Cells(amount)
    self.cell_groups[name] = g_cells
    return g_cells

  def initialize_cells(self):
    self.session.run(tf.global_variables_initializer())

    for monitor_key in self.monitors:
      self.monitors[monitor_key] =\
        self.get_group_cells_state(monitor_key[0], monitor_key[1])

  def close(self):
    self.session.close()

  def run(self,timesteps: int = 10):
    for step in range(timesteps):
      for group_key in self.cell_groups:
        self.session.run(self.cell_groups[group_key].internal_connections)

      for monitor_key in self.monitors:
        self.monitors[monitor_key] = np.vstack((self.monitors[monitor_key],\
          self.session.run(self.cell_groups[monitor_key[0]].states[monitor_key[1]])))
      utils.progressbar(step+1, timesteps)

  def run_step(self):
    for group_key in self.cell_groups:
      self.session.run(self.cell_groups[group_key].internal_connections)

    for monitor_key in self.monitors:
      self.monitors[monitor_key] = np.vstack((self.monitors[monitor_key],\
        self.session.run(self.cell_groups[monitor_key[0]].states[monitor_key[1]])))

  def get_group_cells_state(self, group_cells_name, state_name):
    group_cells_name_exists = group_cells_name in self.cell_groups
    if group_cells_name_exists:
      state_name_exists = state_name in self.cell_groups[group_cells_name].states
      if state_name_exists:
        state = self.cell_groups[group_cells_name].states[state_name]
      else:
        print("Warning: state_name for state does not exist.")
    else:
      print("Warning: group_cells_name for group_cells does not exist.")
    return self.session.run(state)

  def add_monitor(self, group_cells_name, state_name):
    group_cells_name_exists = group_cells_name in self.cell_groups
    if group_cells_name_exists:
      state_name_exists = state_name in self.cell_groups[group_cells_name].states
      if state_name_exists:
        self.monitors[(group_cells_name,state_name)] = None
      else:
        print("Warning: state_name for state does not exist.")
    else:
      print("Warning: group_cells_name for group_cells does not exist.")

  def get_monitor(self, group_cells_name, state_name):
    group_cells_name_exists = group_cells_name in self.cell_groups
    if group_cells_name_exists:
      state_name_exists = state_name in self.cell_groups[group_cells_name].states
      if state_name_exists:
        monitor = self.monitors[(group_cells_name,state_name)]
      else:
        print("Warning: state_name for state does not exist.")
    else:
      print("Warning: group_cells_name for group_cells does not exist.")

    return monitor