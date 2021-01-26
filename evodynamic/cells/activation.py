""" Activation functions for Cells """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def game_of_life_func(count_neighbors, previous_state):
  born_cells_op = tf.equal(count_neighbors, 3)
  kill_cells_sub_op = tf.less(count_neighbors, 2)
  kill_cells_over_op = tf.greater(count_neighbors, 3)

  kill_cells_op = tf.logical_or(kill_cells_sub_op, kill_cells_over_op)

  update_kill_op = tf.where(kill_cells_op, tf.zeros(tf.shape(previous_state), dtype=tf.float64), previous_state)

  # Return update_born_op
  return tf.where(born_cells_op, tf.ones(tf.shape(previous_state), dtype=tf.float64), update_kill_op)

def rule_binary_ca_1d_width3_func(pattern, previous_state, rule):
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

def sigmoid(x, args):
  return tf.sigmoid(x)

def relu(x, args):
  return tf.nn.relu(x)

def stochastic_sigmoid(x, args):
  shape_x = tf.shape(x)
  prob = tf.sigmoid(x)
  prob_mask = tf.less_equal(tf.random.uniform(shape_x, dtype=tf.float64), prob)
  return tf.where(prob_mask,\
                 tf.ones(shape_x, dtype=tf.float64),\
                 tf.zeros(shape_x, dtype=tf.float64))

def _stochastic_prob(prob_mean, prob_std):
  prob_mean_mod = tf.math.log(tf.div_no_nan(prob_mean, 1-prob_mean))
  #sample_random_normal = tf.random.normal([], prob_mean_mod, prob_std, seed=1)
  sample_random_normal = tf.random.normal([], prob_mean_mod, prob_std)
  prob = tf.divide(1, 1+tf.math.exp(-sample_random_normal))
  return prob

def stochastic_prob(prob_mean, prob_std):
  prob = tf.cond(tf.math.logical_and(tf.math.not_equal(prob_mean, 0),
                                     tf.math.not_equal(prob_mean, 1)),
                 true_fn=lambda: _stochastic_prob(prob_mean, prob_std),
                 false_fn=lambda: prob_mean)

  return prob

def rule_binary_soc_sca_1d_width3_func(pattern, previous_state, prob_list):
  shape_previous_state = tf.shape(previous_state)

  pattern_0_op = tf.equal(pattern, 0)
  pattern_1_op = tf.equal(pattern, 1)
  pattern_2_op = tf.equal(pattern, 2)
  pattern_3_op = tf.equal(pattern, 3)
  pattern_4_op = tf.equal(pattern, 4)
  pattern_5_op = tf.equal(pattern, 5)
  pattern_6_op = tf.equal(pattern, 6)
  pattern_7_op = tf.equal(pattern, 7)

  prob_0 = stochastic_prob(prob_list[0], prob_list[8])
  prob_1 = stochastic_prob(prob_list[1], prob_list[8])
  prob_2 = stochastic_prob(prob_list[2], prob_list[8])
  prob_3 = stochastic_prob(prob_list[3], prob_list[8])
  prob_4 = stochastic_prob(prob_list[4], prob_list[8])
  prob_5 = stochastic_prob(prob_list[5], prob_list[8])
  prob_6 = stochastic_prob(prob_list[6], prob_list[8])
  prob_7 = stochastic_prob(prob_list[7], prob_list[8])


  prob_new_state_0 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_0)

  prob_new_state_1 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_1)

  prob_new_state_2 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_2)

  prob_new_state_3 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_3)

  prob_new_state_4 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_4)

  prob_new_state_5 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_5)

  prob_new_state_6 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_6)

  prob_new_state_7 = tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               prob_7)

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
  update_7_op = tf.where(pattern_7_op, new_state_pattern_7, update_6_op)

  #return update_7_op
  new_state_op =  tf.cast(tf.less_equal(tf.random.uniform(shape_previous_state),\
                                               0.5), tf.float64)

  return tf.cond(tf.equal(tf.reduce_mean(previous_state), 1.0), lambda: new_state_op, lambda: update_7_op)


def integrate_and_fire(potential_change, spike_in, potential, threshold, potential_decay):
  shape_potential = tf.shape(potential)
  potential_update_op_1 = tf.add(potential, potential_change)
  has_spike_op = tf.greater(potential_update_op_1, threshold)

  potential_update_op_2 = tf.where(has_spike_op,
                                   tf.zeros(shape_potential, dtype=tf.float64),
                                   tf.subtract(potential_update_op_1, tf.multiply(potential_update_op_1, potential_decay)))

  potential_update_op_3 = tf.assign(potential, potential_update_op_2)
  return tf.cast(tf.logical_and(tf.equal(potential_update_op_3, 0), has_spike_op), tf.float64)





