""" Monitors """
import numpy as np

class Monitor(object):
  def __init__(self, experiment, group_cells_name, state_name, duration=None) -> None:
    """
    Monitor constructor.

    A monitor saves the state of a group of cells as a Numpy array.

    Parameters
    ----------
    experiment : object
        Experiment object.
    group_cells_name : str
        Name of the group of cells.
    state_name : str
        Number of time steps to be saved in memory.
    duration : int, optional
        Maximum number of time steps to store. If none, then there is no limit.

    Returns
    -------
    out : object
        New object of class Monitor.
    """
    group_cells_name_exists = group_cells_name in experiment.cell_groups
    assert group_cells_name_exists, "Error: group_cells_name for group_cells does not exist."

    state_name_exists = state_name in experiment.cell_groups[group_cells_name].states
    assert state_name_exists, "Error: state_name for state does not exist."

    self.experiment = experiment
    self.group_cells_name = group_cells_name
    self.state_name = state_name
    self.state = experiment.cell_groups[group_cells_name].states[state_name]
    self.group_cells_size = experiment.cell_groups[group_cells_name].amount
    self.batch_size = experiment.batch_size
    self.timesteps = duration
    self.timestep_record = 0
    self.state_record = None

  def initialize(self):
    """
    Initializes the monitor.
    """
    if self.timesteps is None:
      self.state_record = np.expand_dims(self.experiment.session.run(self.state), 0)
    else:
      self.state_record = np.zeros((self.timesteps, self.group_cells_size, self.batch_size))
      self.state_record[0] = self.experiment.session.run(self.state)
      self.timestep_record = 1

  def record(self):
    """
    Record the state value in the current time step.
    """
    if self.timesteps is None:
      self.state_record = np.vstack((self.state_record,\
                                     np.expand_dims(self.experiment.session.run(self.state),0)))
    else:
      if self.timestep_record < self.timesteps:
        self.state_record[self.timestep_record] =\
          self.experiment.session.run(self.state)
      else:
        self.state_record = np.vstack((self.state_record[1:],\
                                     np.expand_dims(self.experiment.session.run(self.state),0)))

      self.timestep_record = self.timestep_record+1

  def get(self):
    """
    Returns the recording of the state as a Numpy array.
    """
    return self.state_record