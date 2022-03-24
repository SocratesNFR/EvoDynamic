""" Activation functions for Cells """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def game_of_life_func(count_neighbors, previous_state):
  """
  Activation function for Conway's Game of Life.

  Parameters
  ----------
  count_neighbors : Tensor
      Result of the matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  previous_state : Tensor
      State of a binary state of a group of cells before being updated.

  Returns
  -------
  out : Tensor
      Tensor containing the operation for changing the binary states of a group
      of cells as the Conway's Game of Life.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(ca_binary_state,ca_binary_state,
                                               game_of_life_func,ca_conn)
  """
  born_cells_op = tf.equal(count_neighbors, 3)
  kill_cells_sub_op = tf.less(count_neighbors, 2)
  kill_cells_over_op = tf.greater(count_neighbors, 3)

  kill_cells_op = tf.logical_or(kill_cells_sub_op, kill_cells_over_op)

  update_kill_op = tf.where(kill_cells_op, tf.zeros(tf.shape(previous_state), dtype=tf.float64), previous_state)

  # Return update_born_op
  return tf.where(born_cells_op, tf.ones(tf.shape(previous_state), dtype=tf.float64), update_kill_op)

def life_like_func(count_neighbors, previous_state, born_list, keep_list):
  """
  Activation function for life-like 2D cellular automaton.

  Parameters
  ----------
  count_neighbors : Tensor
      Result of the matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  previous_state : Tensor
      State of a binary state of a group of cells before being updated.
  born_list : list
      List with numbers to born a cell.
  keep_list : list
      List with numbers to keep a cell.
  n_neighbors : int
      Number of neighbors.

  Returns
  -------
  out : Tensor
      Tensor containing the operation for changing the binary states of a group
      of cells as the Conway's Game of Life.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(ca_binary_state,ca_binary_state,
                                               life_like_func,ca_conn)
  """
  born_cells_list_op = [tf.equal(count_neighbors, b) for b in born_list]
  born_cells_op = tf.reduce_any(tf.concat(born_cells_list_op, 1), 1)

  kill_cells_list_op = [tf.not_equal(count_neighbors, k) for k in keep_list]
  kill_cells_op = tf.reduce_all(tf.concat(kill_cells_list_op, 1), 1)

  update_kill_op = tf.where(kill_cells_op, tf.zeros(tf.shape(previous_state), dtype=tf.float64), previous_state)

  # Return update_born_op
  return tf.where(born_cells_op, tf.ones(tf.shape(previous_state), dtype=tf.float64), update_kill_op)


def rule_binary_ca_1d_width3_func(pattern, previous_state, rule):
  """
  Activation function for 3-neighbor Elementary Cellular Automaton.

  Parameters
  ----------
  pattern : Tensor
      Result of the matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  previous_state : Tensor
      State of a binary state of a group of cells before being updated.
  rule: int between 0 and 255
      rule number for the 3-neighbor Elementary Cellular Automaton.

  Returns
  -------
  out : Tensor
      Tensor containing the operation for changing the binary states of a group
      of cells as expressed by the chosen rule of the elementary celullar
      automaton.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object. The activation functions with more the two parameters
  must have the extra parameters passed through the parameter 'fargs_list' of
  'evodynamic.connection.WeightedConnection'.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(ca_binary_state,ca_binary_state,
                                               rule_binary_ca_1d_width3_func,
                                               ca_conn,
                                               fargs_list=[(rule,)])
  """
  pattern_0_op = tf.equal(pattern, 0)
  pattern_1_op = tf.equal(pattern, 1)
  pattern_2_op = tf.equal(pattern, 2)
  pattern_3_op = tf.equal(pattern, 3)
  pattern_4_op = tf.equal(pattern, 4)
  pattern_5_op = tf.equal(pattern, 5)
  pattern_6_op = tf.equal(pattern, 6)
  pattern_7_op = tf.equal(pattern, 7)

  new_state_pattern_0 = tf.ones(tf.shape(previous_state), dtype=tf.float64)\
                        if (rule & (1<<0)) != 0 else\
                        tf.zeros(tf.shape(previous_state), dtype=tf.float64)
  new_state_pattern_1 = tf.ones(tf.shape(previous_state), dtype=tf.float64)\
                        if (rule & (1<<1)) != 0 else\
                        tf.zeros(tf.shape(previous_state), dtype=tf.float64)
  new_state_pattern_2 = tf.ones(tf.shape(previous_state), dtype=tf.float64)\
                        if (rule & (1<<2)) != 0 else\
                        tf.zeros(tf.shape(previous_state), dtype=tf.float64)
  new_state_pattern_3 = tf.ones(tf.shape(previous_state), dtype=tf.float64)\
                        if (rule & (1<<3)) != 0 else\
                        tf.zeros(tf.shape(previous_state), dtype=tf.float64)
  new_state_pattern_4 = tf.ones(tf.shape(previous_state), dtype=tf.float64)\
                        if (rule & (1<<4)) != 0 else\
                        tf.zeros(tf.shape(previous_state), dtype=tf.float64)
  new_state_pattern_5 = tf.ones(tf.shape(previous_state), dtype=tf.float64)\
                        if (rule & (1<<5)) != 0 else\
                        tf.zeros(tf.shape(previous_state), dtype=tf.float64)
  new_state_pattern_6 = tf.ones(tf.shape(previous_state), dtype=tf.float64)\
                        if (rule & (1<<6)) != 0 else\
                        tf.zeros(tf.shape(previous_state), dtype=tf.float64)
  new_state_pattern_7 = tf.ones(tf.shape(previous_state), dtype=tf.float64)\
                        if (rule & (1<<7)) != 0 else\
                        tf.zeros(tf.shape(previous_state), dtype=tf.float64)

  update_0_op = tf.where(pattern_0_op, new_state_pattern_0, previous_state)
  update_1_op = tf.where(pattern_1_op, new_state_pattern_1, update_0_op)
  update_2_op = tf.where(pattern_2_op, new_state_pattern_2, update_1_op)
  update_3_op = tf.where(pattern_3_op, new_state_pattern_3, update_2_op)
  update_4_op = tf.where(pattern_4_op, new_state_pattern_4, update_3_op)
  update_5_op = tf.where(pattern_5_op, new_state_pattern_5, update_4_op)
  update_6_op = tf.where(pattern_6_op, new_state_pattern_6, update_5_op)

  # Return update_7_op
  return tf.where(pattern_7_op, new_state_pattern_7, update_6_op)

def rule_binary_sca_1d_width3_func(pattern, previous_state, prob_list):
  """
  Activation function for 3-neighbor Stochastic Elementary Cellular Automaton.

  Parameters
  ----------
  pattern : Tensor
      Result of the matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  previous_state : Tensor
      State of a binary state of a group of cells before being updated.
  prob_list: list of 8 float numbers between 0 and 1
      Probability list for the 8 different neighborhood patterns.

  Returns
  -------
  out : Tensor
      Tensor containing the operation for changing the binary states of a group
      of cells.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object. The activation functions with more the two parameters
  must have the extra parameters passed through the parameter 'fargs_list' of
  'evodynamic.connection.WeightedConnection'.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(ca_binary_state,ca_binary_state,
                                               rule_binary_sca_1d_width3_func,
                                               ca_conn,
                                               fargs_list=[(prob_list,)])
  """
  shape_previous_state = tf.shape(previous_state)
  pattern_0_op = tf.equal(pattern, 0)
  pattern_1_op = tf.equal(pattern, 1)
  pattern_2_op = tf.equal(pattern, 2)
  pattern_3_op = tf.equal(pattern, 3)
  pattern_4_op = tf.equal(pattern, 4)
  pattern_5_op = tf.equal(pattern, 5)
  pattern_6_op = tf.equal(pattern, 6)
  pattern_7_op = tf.equal(pattern, 7)

  prob_new_state_0 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_list[0])

  prob_new_state_1 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_list[1])

  prob_new_state_2 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_list[2])

  prob_new_state_3 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_list[3])

  prob_new_state_4 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_list[4])

  prob_new_state_5 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_list[5])

  prob_new_state_6 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_list[6])

  prob_new_state_7 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_list[7])

  new_state_pattern_0 = tf.where(prob_new_state_0,\
                                 tf.ones(shape_previous_state, dtype=tf.float64),\
                                 tf.zeros(shape_previous_state, dtype=tf.float64))

  new_state_pattern_1 = tf.where(prob_new_state_1,\
                                 tf.ones(shape_previous_state, dtype=tf.float64),\
                                 tf.zeros(shape_previous_state, dtype=tf.float64))

  new_state_pattern_2 = tf.where(prob_new_state_2,\
                                 tf.ones(shape_previous_state, dtype=tf.float64),\
                                 tf.zeros(shape_previous_state, dtype=tf.float64))

  new_state_pattern_3 = tf.where(prob_new_state_3,\
                                 tf.ones(shape_previous_state, dtype=tf.float64),\
                                 tf.zeros(shape_previous_state, dtype=tf.float64))

  new_state_pattern_4 = tf.where(prob_new_state_4,\
                                 tf.ones(shape_previous_state, dtype=tf.float64),\
                                 tf.zeros(shape_previous_state, dtype=tf.float64))

  new_state_pattern_5 = tf.where(prob_new_state_5,\
                                 tf.ones(shape_previous_state, dtype=tf.float64),\
                                 tf.zeros(shape_previous_state, dtype=tf.float64))

  new_state_pattern_6 = tf.where(prob_new_state_6,\
                                 tf.ones(shape_previous_state, dtype=tf.float64),\
                                 tf.zeros(shape_previous_state, dtype=tf.float64))

  new_state_pattern_7 = tf.where(prob_new_state_7,\
                                 tf.ones(shape_previous_state, dtype=tf.float64),\
                                 tf.zeros(shape_previous_state, dtype=tf.float64))

  update_0_op = tf.where(pattern_0_op, new_state_pattern_0, previous_state)
  update_1_op = tf.where(pattern_1_op, new_state_pattern_1, update_0_op)
  update_2_op = tf.where(pattern_2_op, new_state_pattern_2, update_1_op)
  update_3_op = tf.where(pattern_3_op, new_state_pattern_3, update_2_op)
  update_4_op = tf.where(pattern_4_op, new_state_pattern_4, update_3_op)
  update_5_op = tf.where(pattern_5_op, new_state_pattern_5, update_4_op)
  update_6_op = tf.where(pattern_6_op, new_state_pattern_6, update_5_op)

  # Return update_7_op
  return tf.where(pattern_7_op, new_state_pattern_7, update_6_op)

def sigmoid(x, empty_parameter):
  """
  Sigmoid activation function. It works as a wrapped for EvoDynamic functions.

  Parameters
  ----------
  x : Tensor
      Input value or the result of the matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  empty_parameter : Tensor
      Empty parameter for matching the necessary number of parameters in case
      of using evodynamic.connection.WeightedConnection.

  Returns
  -------
  out : Tensor
      Activation function result.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(ann_layer_1,ann_layer_2,
                                               sigmoid,l1_l2_connection)
  """
  return tf.sigmoid(x)

def relu(x, empty_parameter):
  """
  Rectified linear unit activation function. It works as a wrapped for
  EvoDynamic functions.

  Parameters
  ----------
  x : Tensor
      Input value or the result of the matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  empty_parameter : Tensor
      Empty parameter for matching the necessary number of parameters in case
      of using evodynamic.connection.WeightedConnection.

  Returns
  -------
  out : Tensor
      Activation function result.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(ann_layer_1,ann_layer_2,
                                               relu,l1_l2_connection)
  """
  return tf.nn.relu(x)

def stochastic_sigmoid(x, empty_parameter):
  """
  Stochastic sigmoid activation function. It uses the sigmoid result as the
  probability of selecting a binary state.

  Parameters
  ----------
  x : Tensor
      Input value or the result of the matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  empty_parameter : Tensor
      Empty parameter for matching the necessary number of parameters in case
      of using evodynamic.connection.WeightedConnection.

  Returns
  -------
  out : Tensor
      Activation function result.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(ann_layer_1,ann_layer_2,
                                               stochastic_sigmoid,
                                               l1_l2_connection)
  """
  shape_x = tf.shape(x)
  prob = tf.sigmoid(x)
  prob_mask = tf.less_equal(tf.random.uniform(shape_x, dtype=tf.float64), prob)
  return tf.where(prob_mask,\
                 tf.ones(shape_x, dtype=tf.float64),\
                 tf.zeros(shape_x, dtype=tf.float64))

def _stochastic_prob(prob_mean, prob_std):
  """
  Private stochastic probability generates a probability from a defined
  normal distribution. This function is used by
  'stochastic_prob'.

  Parameters
  ----------
  prob_mean : Tensor
      Mean of the normal distribution.
  prob_std : Tensor
      Standard deviation of the normal distribution.

  Returns
  -------
  out : Tensor
      Generated probability with value between 0 and 1.
  """
  prob_mean_mod = tf.math.log(tf.div_no_nan(prob_mean, 1-prob_mean))
  #sample_random_normal = tf.random.normal([], prob_mean_mod, prob_std, seed=1)
  sample_random_normal = tf.random.normal([], prob_mean_mod, prob_std)
  prob = tf.divide(1, 1+tf.math.exp(-sample_random_normal))
  return prob

def stochastic_prob(prob_mean, prob_std):
  """
  Stochastic probability generates a probability from a defined
  normal distribution. If prob_mean is 0 or 1, the value is returned unchanged.
  This function is used by 'rule_binary_soc_sca_1d_width3_func'.

  Parameters
  ----------
  prob_mean : Tensor
      Mean of the normal distribution.
  prob_std : Tensor
      Standard deviation of the normal distribution.

  Returns
  -------
  out : Tensor
      Generated probability with value between 0 and 1.
  """
  prob = tf.cond(tf.math.logical_and(tf.math.not_equal(prob_mean, 0),
                                     tf.math.not_equal(prob_mean, 1)),
                 true_fn=lambda: _stochastic_prob(prob_mean, prob_std),
                 false_fn=lambda: prob_mean)

  return prob

def rule_binary_soc_sca_1d_width3_func(pattern, previous_state, prob_list):
  """
  Activation function for 3-neighbor Stochastic Elementary Cellular Automaton
  for analysis of self-organized criticality.
  Similar to 'rule_binary_sca_1d_width3_func', but if all states are 1, then
  initialize randomly the state.

  Parameters
  ----------
  pattern : Tensor
      Result of the matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  previous_state : Tensor
      State of a binary state of a group of cells before being updated.
  prob_list: list of 8 float numbers between 0 and 1
      Probability list for the 8 different neighborhood patterns.

  Returns
  -------
  out : Tensor
      Tensor containing the operation for changing the binary states of a group
      of cells.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object. The activation functions with more the two parameters
  must have the extra parameters passed through the parameter 'fargs_list' of
  'evodynamic.connection.WeightedConnection'.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(ca_binary_state,ca_binary_state,
                                               rule_binary_soc_sca_1d_width3_func,
                                               ca_conn,
                                               fargs_list=[(prob_list,)])
  """
  shape_previous_state = tf.shape(previous_state)

  prob_0 = stochastic_prob(prob_list[0], prob_list[8])
  prob_1 = stochastic_prob(prob_list[1], prob_list[8])
  prob_2 = stochastic_prob(prob_list[2], prob_list[8])
  prob_3 = stochastic_prob(prob_list[3], prob_list[8])
  prob_4 = stochastic_prob(prob_list[4], prob_list[8])
  prob_5 = stochastic_prob(prob_list[5], prob_list[8])
  prob_6 = stochastic_prob(prob_list[6], prob_list[8])
  prob_7 = stochastic_prob(prob_list[7], prob_list[8])

  new_prob_list = [prob_0,prob_1,prob_2,prob_3,prob_4,prob_5,prob_6,prob_7]

  update_7_op = rule_binary_sca_1d_width3_func(pattern, previous_state,
                                               new_prob_list)

  #return update_7_op
  new_state_op =  tf.cast(tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               0.5), tf.float64)

  return tf.cond(tf.equal(tf.reduce_mean(previous_state), 1.0), lambda: new_state_op, lambda: update_7_op)

def integrate_and_fire(potential_change, spike_in, potential, threshold, potential_decay):
  """
  Activation function for spiking neural networks with neuron model
  integrate and fire.

  Parameters
  ----------
  potential_change : Tensor
      Change of the membrane potential, which comes from the result of the
      matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  spike_in : Tensor
      Input spikes.
  potential : Tensor
      Membrane potential state from the group of cells defined by
      the parameter 'to_group_state' in evodynamic.connection.WeightedConnection.
  threshold : float
      Threshold value that triggers the production of an output spike if the
      value of the membrane potential in higher than the threshold.
  potential_decay : float
      Decay value of the membrane potential.

  Returns
  -------
  out : Tensor
      Tensor containing the operation for the output spikes.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object. The activation functions with more the two parameters
  must have the extra parameters passed through the parameter 'fargs_list' of
  'evodynamic.connection.WeightedConnection'.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(spike_state_layer_1,
                                               spike_state_layer_2, act.integrate_and_fire,
                                               l1_l2_connection,
                                               fargs_list=[(potential,threshold,
                                                            potential_decay)])
  """
  shape_potential = tf.shape(potential)
  potential_update_op_1 = tf.add(potential, potential_change)
  has_spike_op = tf.greater(potential, threshold)

  potential_update_op_2 = tf.where(has_spike_op,
                                   tf.zeros(shape_potential, dtype=tf.float64),
                                   tf.subtract(potential_update_op_1, tf.multiply(potential_update_op_1, potential_decay)))

  potential_update_op_3 = tf.assign(potential, potential_update_op_2)
  return tf.cast(has_spike_op, tf.float64) + (0*potential_update_op_3)


def izhikevich(potential_change, spike_in, potential, recovery, a, b, c, d, dt):
  """
  Activation function for spiking neural networks with Izhikevich neuron model.

  Parameters
  ----------
  potential_change : Tensor
      Change of the membrane potential, which comes from the result of the
      matrix multiplication that happens in
      evodynamic.connection.WeightedConnection
  spike_in : Tensor
      Input spikes.
  potential : Tensor
      Membrane potential state from the group of cells defined by
      the parameter 'to_group_state' in evodynamic.connection.WeightedConnection.
  recovery : float
      Membrane recovery state from the group of cells defined by
      the parameter 'to_group_state' in evodynamic.connection.WeightedConnection.
  a : float
      Parameter 'a' of Izhikevich model.
  b : float
      Parameter 'b' of Izhikevich model.
  c : float
      Parameter 'c' of Izhikevich model.
  d : float
      Parameter 'd' of Izhikevich model.
  dt : float
      Value in milliseconds of the time-steps in simulation.

  Returns
  -------
  out : Tensor
      Tensor containing the operation for the output spikes.

  Notes
  -----
  The values of the first two parameters come from operations in
  the Connection object. The activation functions with more the two parameters
  must have the extra parameters passed through the parameter 'fargs_list' of
  'evodynamic.connection.WeightedConnection'.

  Examples
  --------
  >>> evodynamic.connection.WeightedConnection(spike_state_layer_1,
                                               spike_state_layer_2, act.izhikevich,
                                               l1_l2_connection,
                                               fargs_list=[(potential,recovery,
                                                            a,b,c,d,dt)])
  """
  shape_potential = tf.shape(potential)
  has_spike_op = tf.greater(potential, 30.0)

  potential_update_op_1 = tf.pow(potential, 2.0)
  potential_update_op_1 = tf.multiply(potential_update_op_1, 0.04)
  potential_update_op_1 = tf.add(potential_update_op_1, tf.multiply(potential, 5.0))
  potential_update_op_1 = tf.add(potential_update_op_1, 140.0)
  potential_update_op_1 = tf.subtract(potential_update_op_1, recovery)
  potential_update_op_1 = tf.add(potential_update_op_1, potential_change)
  potential_update_op_1 = tf.multiply(potential_update_op_1, dt)
  potential_update_op_1 = tf.add(potential_update_op_1, potential)

  recovery_update_op_1 = tf.multiply(potential, b)
  recovery_update_op_1 = tf.subtract(recovery_update_op_1, recovery)
  recovery_update_op_1 = tf.multiply(recovery_update_op_1, a)
  recovery_update_op_1 = tf.multiply(recovery_update_op_1, dt)
  recovery_update_op_1 = tf.add(recovery_update_op_1, recovery)

  potential_update_op_2 = tf.where(has_spike_op,
                                   tf.cast(tf.fill(shape_potential, c), tf.float64),
                                   potential_update_op_1)

  recovery_update_op_2 = tf.where(has_spike_op,
                                   tf.add(recovery, d),
                                   recovery_update_op_1)

  potential_update_op_3 = tf.assign(potential, potential_update_op_2)
  recovery_update_op_3 = tf.assign(recovery, recovery_update_op_2)
  return tf.cast(has_spike_op, tf.float64) + (0*potential_update_op_3) + (0*recovery_update_op_3)
