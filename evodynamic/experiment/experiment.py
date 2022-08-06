""" Experiment """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from . import monitor
from . import memory
from .. import cells
from .. import utils

class Experiment(object):
  def __init__(self, dt: float = 1.0, input_start=0, input_delay=0,\
               training_start=0, training_delay=0, batch_size=1,\
               reset_cells_after_train=False, reset_memories_after_train=False,
               input_delay_until_train=False) -> None:
    """
    Experiment constructor.

    TensorFlow graph and its operations are controlled in this class.

    Parameters
    ----------
    dt : float
        Size of the time step.
    input_start : int
        Time step for starting adding input.
    input_delay : int
        Delay for adding the next input.
    training_start : int
        Time step for starting the training.
    training_delay : int
        Delay for training.
    batch_size : int
        Batch size.
    reset_cells_after_train : Boolean
        Reset cells after the time step with training.
    reset_memories_after_train : Boolean
        Reset memories after the time step with training.
    input_delay_until_train : Boolean
        Next input happens after the time step with training.

    Returns
    -------
    out : object
        New object of class Experiment.
    """
    tf.reset_default_graph()
    self.dt = dt
    # Dictionary of cell groups
    self.cell_groups = {}
    # Dictionary of connections
    self.connections = {}
    self.connection_list = []
    # Dictionary of trainable connections
    self.trainable_connections = {}
    # List of operations for the connections
    self.connection_ops = []
    self.input_name_list = []
    self.input_placeholder_list = []
    self.input_ops = []
    self.desired_output_name_list = []
    self.desired_output_placeholder_list = []
    self.train_ops = []
    # Dictionary of monitors
    self.monitors = {}
    # Initializing tf.Session
    self.session = tf.Session()
    # Dictionary of memories
    self.memories = {}
    # List of operations for the memories
    self.memory_ops = []
    self.step_counter = 0
    self.input_start = input_start
    self.input_delay = input_delay
    self.input_delay_until_train = input_delay_until_train
    self.input_tracker = -1
    self.training_start = training_start
    self.training_delay = training_delay
    self.training_tracker = -1
    # Dictionary of experiment outputs
    self.experiment_output = {}
    self.has_input = tf.placeholder(tf.bool, shape=())
    self.training_loss_op = []
    self.training_loss = np.nan
    self.batch_size = batch_size
    self.reset_cells_after_train = reset_cells_after_train
    self.reset_memories_after_train = reset_memories_after_train
    self.next_step_after_train = False
    self.training_input = None
    # Dictionary of training outputs
    self.training_output = {}

  def add_input(self, dtype, shape, name):
    """
    Add input placeholder to the experiment.

    Parameters
    ----------
    dtype : dtype
        Type of the input data.
    shape : tuple
        Tuple with the shape of the input.
    name : str
        Name of the input.

    Returns
    -------
    out : Tensor
        Tensor to be used to feed input values to the experiment.
    """
    shape_with_batch = list(shape)
    shape_with_batch.insert(1,self.batch_size)
    input_placeholder = tf.placeholder(dtype, shape=shape_with_batch, name=name)
    self.input_name_list.append(name)
    self.input_placeholder_list.append(input_placeholder)
    return input_placeholder

  def add_desired_output(self, dtype, shape, name):
    """
    Add desired output placeholder to the experiment.

    Parameters
    ----------
    dtype : dtype
        Type of the desired output data.
    shape : tuple
        Tuple with the shape of the desired output.
    name : str
        Name of the desired output.

    Returns
    -------
    out : Tensor
        Tensor to be used to feed desired output values to the experiment.
    """
    shape_with_batch = list(shape)
    shape_with_batch.insert(1,self.batch_size)
    desired_output_placeholder = tf.placeholder(dtype, shape=shape_with_batch, name=name)
    self.desired_output_name_list.append(name)
    self.desired_output_placeholder_list.append(desired_output_placeholder)
    return desired_output_placeholder

  def add_group_cells(self, name, amount, virtual_shape=None):
    """
    Create and add cells object to the experiment.

    Parameters
    ----------
    name : str
        Name of the group of cells.
    amount : int
        Amount of cells in the group.
    virtual_shape : tuple or list
        Shape of the group of cells if they had one.

    Returns
    -------
    out : object
        Object from the class evodynamic.cells.Cells.
    """
    g_cells = cells.Cells(amount, self.batch_size, virtual_shape)
    self.cell_groups[name] = g_cells
    return g_cells

  def update_experiment_output(self, new_connection):
    """
    Private function for updating the output of the experiment.

    Parameters
    ----------
    new_connection : object
        Connection object recently added, so it becomes the output of the
        experiment.
    """
    if new_connection.from_group_state in self.experiment_output and\
      new_connection.to_group_state not in self.experiment_output:
      del self.experiment_output[new_connection.from_group_state]
    self.experiment_output[new_connection.to_group_state] = new_connection

  def update_training_output(self, new_connection):
    """
    Private function for updating the output of a trainable connection in the
    experiment.

    Parameters
    ----------
    new_connection : object
        Connection object recently added, so it becomes the trainable output of
        the experiment.
    """
    if new_connection.from_group_state in self.training_output and\
      new_connection.to_group_state not in self.training_output:
      del self.training_output[new_connection.from_group_state]
    self.training_output[new_connection.to_group_state] = new_connection


  def add_state_memory(self, state, memory_size):
    """
    Create and add state memory object to the experiment.

    Parameters
    ----------
    state : Tensor
        Tensor with the state of a group of cells.
    memory_size : int
        Amount of time steps for saving the state.

    Returns
    -------
    out : Tensor
        Tensor variable for the memory.
    """
    state_memory = memory.Memory(self,state,memory_size)
    self.memories[state] = state_memory
    memory_op = state_memory.get_op()
    self.memory_ops.append(memory_op)
    self.connection_list.insert(0,state_memory)
    self.update_experiment_output(state_memory)
    return memory_op

  def add_connection(self, name, connection):
    """
    Add connection object to the experiment.

    Parameters
    ----------
    name : str
        Name of the connection.
    connection : object
        Connection object.

    Returns
    -------
    out : Tensor
        Tensor variable for the connection.
    """
    connection.set_experiment(self)
    self.connections[name] = connection
    self.connection_list.insert(0,connection)
    self.connection_ops.append(connection.list_ops)
    self.update_experiment_output(connection)
    if connection.from_group_state.name.split(":")[0] in self.input_name_list: # if input
      self.input_ops.append(connection.list_ops)
    else:
      self.connection_ops.append(connection.list_ops)
    return connection.assign_output

  def add_trainable_connection(self, name, connection):
    """
    Add connection object for training to the experiment.

    Parameters
    ----------
    name : str
        Name of the connection.
    connection : object
        Connection object.

    Returns
    -------
    out : Tensor
        Tensor variable for the connection.
    """
    if self.training_input == None:
      self.training_input = connection.from_group_state
      connection.set_experiment(self, is_first_training_connection=True)
    else:
      connection.set_experiment(self, is_first_training_connection=False)
    self.trainable_connections[name] = connection
    self.connections[name] = connection
    self.connection_list.insert(0,connection)
    self.connection_ops.append(connection.list_ops)
    self.update_training_output(connection)

    if connection.from_group_state.name.split(":")[0] in self.input_name_list: # if input
      self.input_ops.append(connection.list_ops)
    else:
      self.connection_ops.append(connection.list_ops)
    return connection.assign_output

  def initialize_cells(self):
    """
    Initialize the states of the cells and monitors.
    """
    self.session.run(tf.global_variables_initializer())
    for monitor_key in self.monitors:
      self.monitors[monitor_key].initialize()

  def reset_cell_states(self):
    """
    Reset the states of the cells.
    """
    for cell_group_key in self.cell_groups:
      for state_key in self.cell_groups[cell_group_key].states:
        state = self.cell_groups[cell_group_key].states[state_key]
        #self.session.run(tf.assign(state, tf.zeros_like(state)))
        self.session.run(state.initializer)

  def set_training(self, loss, learning_rate, optimizer="adam"):
    """
    Set the parameters for training.

    Parameters
    ----------
    loss : Tensor
        Tensor for the loss.
    learning_rate : float
        Learning rate.
    optimizer : str, {'adam'}
        Optimizer for training.
    """
    model_vars = tf.trainable_variables()
    self.training_loss_op = [loss]
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
    """
    Close tf.Session and experiment.
    """
    self.session.close()

  def is_input_step(self):
    """
    Identifies the time step with input.
    """
    if self.input_delay_until_train and self.input_delay == None:
      is_input_step = self.input_start == self.step_counter or self.next_step_after_train
    else:
      is_input_step = True
      if self.input_delay > 0:
        is_input_step = ((self.step_counter-self.input_start + 1) // (self.input_delay)) > self.input_tracker

    if is_input_step:
      self.input_tracker += 1

    return is_input_step

  def is_training_step(self):
    """
    Identifies the time step with training.
    """
    is_training_step = len(self.train_ops) > 0
    if self.training_delay > 0 and is_training_step:
      is_training_step = ((self.step_counter-self.training_start + 1) // (self.training_delay)) > self.training_tracker

    if is_training_step:
      self.training_tracker += 1
    return is_training_step

  def run(self,timesteps: int = 10):
    """
    Run experiment for a number of time steps.

    Parameters
    ----------
    timesteps : int
        Number of time steps to run.
    """
    for step in range(timesteps-1):
      self.run_step()
      utils.progressbar(step+1, timesteps-1)

  def run_with_input_list(self, timesteps: int, feed_dict_list, testing=False):
    """
    Run experiment for a number of time steps while feeding input and desired
    output placeholders.

    Parameters
    ----------
    timesteps : int
        Number of time steps to run.
    feed_dict_list : list
        List with data for feeding input and desired output placeholders.
    testing : Boolean
        Indicates if it is a testing run. If yes, optimizer will not be applied.
    """
    feed_counter = 0
    for step in range(timesteps-1):
      if self.is_input_step() or self.is_training_step():
        feed_counter += 1

      self.run_step(feed_dict=feed_dict_list[feed_counter], testing=testing)
      utils.progressbar(step+1, timesteps-1)

  #TODO: Test this function
  def run_with_input_generator(self, timesteps: int, generator, testing=False):
    """
    Run experiment for a number of time steps while feeding input and desired
    output placeholders using a data generator function.

    Parameters
    ----------
    timesteps : int
        Number of time steps to run.
    generator : function
        Function that generates data for feeding input and desired output
        placeholders.
    testing : Boolean
        Indicates if it is a testing run. If yes, optimizer will not be applied.
    """
    for step in range(timesteps-1):
      if self.is_input_step() or self.is_training_step():

        feed_dict = generator(self.step_counter)
        self.run_step(feed_dict=feed_dict)
      else:
        self.run_step()
      utils.progressbar(step+1, timesteps-1)

  def run_step(self, feed_dict=None, testing=False):
    """
    Run experiment for one time step.

    Parameters
    ----------
    feed_dict_list : list
        List with data for feeding input and desired output placeholders.
    testing : Boolean
        Indicates if it is a testing run. If yes, optimizer will not be applied.
    """
    if not feed_dict:
      feed_dict = {}

    if self.next_step_after_train and self.reset_cells_after_train:
      self.reset_cell_states()
    if self.next_step_after_train and self.reset_memories_after_train:
      for memory_key in self.memories:
        self.memories[memory_key].reset()

    # After checking that training happened in the previous step, then reset
    # self.next_step_after_train
    self.next_step_after_train = False

    feed_dict[self.has_input] = self.is_input_step()

    experiment_ops = []
    for experiment_output_key in self.experiment_output:
      experiment_ops.append(self.experiment_output[experiment_output_key].assign_output)

    # for memory_key in self.memories:
    #   experiment_ops.append(self.memories[memory_key].update_state_memory())

    self.session.run(experiment_ops,feed_dict=feed_dict)

    # for memory_key in self.memories:
    #   self.memories[memory_key].update_state_memory()

    if self.is_training_step():
      training_ops = []
      for training_output_key in self.training_output:
        training_ops.append(self.training_output[training_output_key].assign_output)

      if testing:
        training_ops += self.training_loss_op
      else:
        training_ops += self.train_ops + self.training_loss_op

      training_result = self.session.run(training_ops,feed_dict=feed_dict)
      self.next_step_after_train = True

      if len(training_result) > 0:
        self.training_loss = training_result[-1]

    for monitor_key in self.monitors:
      self.monitors[monitor_key].record()

    self.step_counter += 1

  def check_group_cells_state(self, group_cells_name, state_name):
    """
    Checks if a state exists in a group of cells.

    Parameters
    ----------
    group_cells_name : str
        Name of group of cells.
    state_name : str
        Name of the state.
    """
    group_cells_name_exists = group_cells_name in self.cell_groups
    assert group_cells_name_exists, "Error: group_cells_name for group_cells does not exist."

    state_name_exists = state_name in self.cell_groups[group_cells_name].states
    assert state_name_exists, "Error: state_name for state does not exist."

  def get_group_cells_state(self, group_cells_name, state_name):
    """
    Gets a state from a group of cells.

    Parameters
    ----------
    group_cells_name : str
        Name of group of cells.
    state_name : str
        Name of the state.
    """
    self.check_group_cells_state(group_cells_name, state_name)

    return self.session.run(self.cell_groups[group_cells_name].states[state_name])

  def add_monitor(self, group_cells_name, state_name, timesteps=None):
    """
    Add monitor of a state to the experiment.

    Parameters
    ----------
    group_cells_name : str
        Name of group of cells.
    state_name : str
        Name of the state.
    timesteps : int
        Number of time steps the monitor saves
    """
    self.check_group_cells_state(group_cells_name, state_name)

    self.monitors[(group_cells_name,state_name)] =\
      monitor.Monitor(self, group_cells_name, state_name, duration=timesteps)

  def get_monitor(self, group_cells_name, state_name):
    """
    Get values of the monitor of a state.

    Parameters
    ----------
    group_cells_name : str
        Name of group of cells.
    state_name : str
        Name of the state.

    Returns
    -------
    out : Numpy array
        Array with the values of the monitor.
    """
    self.check_group_cells_state(group_cells_name, state_name)

    return self.monitors[(group_cells_name,state_name)].get()

  def get_connection(self, conn_name):
    """
    Get connection object from name.

    Parameters
    ----------
    conn_name : str
        Name of the connection.

    Returns
    -------
    out : Object
        Connection object.
    """
    conn_name_exists = conn_name in self.connections
    assert conn_name_exists, "Error: conn_name for connections does not exist."

    return self.connections[conn_name]