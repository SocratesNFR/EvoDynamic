""" Monitors """
import numpy as np

class Monitor(object):
  def __init__(self, experiment, group_cells_name, state_name, duration=None, duration_type="timesteps") -> None:

    group_cells_name_exists = group_cells_name in experiment.cell_groups
    assert group_cells_name_exists, "Error: group_cells_name for group_cells does not exist."

    state_name_exists = state_name in experiment.cell_groups[group_cells_name].states
    assert state_name_exists, "Error: state_name for state does not exist."

    self.experiment = experiment
    self.group_cells_name = group_cells_name
    self.state_name = state_name
    self.state = experiment.cell_groups[group_cells_name].states[state_name]
    self.group_cells_size = experiment.cell_groups[group_cells_name].amount
    self.timesteps = duration
    self.timestep_record = 0
    self.state_record = None

  def initialize(self):
    if self.timesteps is None:
      self.state_record = self.experiment.session.run(self.state)
    else:
      self.state_record = np.zeros((self.timesteps, self.group_cells_size))
      self.state_record[0] = self.experiment.session.run(self.state)
      self.timestep_record = 1

  def record(self):
    if self.timesteps is None:
      self.state_record = np.vstack((self.state_record,\
                                     self.experiment.session.run(self.state)))
    else:
      if self.timestep_record < self.timesteps:
        self.state_record[self.timestep_record] =\
          self.experiment.session.run(self.state)
      else:
        self.state_record = np.vstack((self.state_record[1:],\
                                     self.experiment.session.run(self.state)))

      self.timestep_record = self.timestep_record+1

  def get(self):
    return self.state_record