""" Cellular automata 1D with input - animation """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np
import matplotlib.pyplot as plt

width = 100
timesteps = 400
input_size = width // 2
batch_size = 2

exp = experiment.Experiment(input_start=25,input_delay=24, batch_size=batch_size)
input_ca = exp.add_input(tf.float64, [input_size], "input_ca")
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

fargs_list = [(a,) for a in [204]]

exp.add_connection("input_conn", connection.IndexConnection(input_ca,g_ca_bin,
                                                            np.arange(input_size)))

exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))


exp.add_monitor("g_ca", "g_ca_bin")

exp.initialize_cells()

def input_generator(step):
  return {input_ca: np.zeros((input_size,batch_size)) if ((step // 10) % 2 == 0) else np.ones((input_size,batch_size))}

exp.run_with_input_generator(timesteps, input_generator)

ca_result = exp.get_monitor("g_ca", "g_ca_bin")

fig, axs = plt.subplots(1, batch_size)
for i in range(batch_size):
  axs[i].imshow(ca_result[:,:,i])
plt.show()