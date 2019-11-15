""" Activation functions for Cells """

import tensorflow as tf

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


