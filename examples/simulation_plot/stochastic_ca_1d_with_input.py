""" Stochastic Cellular automata 1D with input - animation """

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

exp = experiment.Experiment(input_start=25,input_delay=24)
input_ca = exp.add_input(tf.float64, [input_size], "input_ca")
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

"""
Best 2
80;4.375510397876782;{'norm_ksdist_res': 0.9613162536023352,
'norm_coef_res': 1.6377095314393233, 'norm_unique_states': 1.0,
'norm_avalanche_pdf_size': 0.9659114597767141, 'norm_linscore_res': 0.8704469695084798,
'norm_R_res': 0.7277920719335422, 'fitness': 4.375510397876782};
[0.39422172047670734, 0.09472197628905793, 0.2394927250526252, 0.4084554505943707, 0.0, 0.7302038703441202, 0.9150343715586952, 1.0]
"""
fargs_list = [( [0.39422172047670734, 0.09472197628905793,\
                 0.2394927250526252, 0.4084554505943707, 0.0,\
                 0.7302038703441202, 0.9150343715586952, 1.0],)]


exp.add_connection("input_conn", connection.IndexConnection(input_ca,g_ca_bin,
                                                            np.arange(input_size)))

exp.add_connection("g_ca_conn",
                     connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                   act.rule_binary_sca_1d_width3_func,
                                                   g_ca_bin_conn, fargs_list=fargs_list))

exp.add_monitor("g_ca", "g_ca_bin")

exp.initialize_cells()

def input_generator(step):
  return {input_ca: np.zeros((input_size,1)) if ((step // 10) % 2 == 0) else np.ones((input_size,1))}

exp.run_with_input_generator(timesteps, input_generator)

ca_result = exp.get_monitor("g_ca", "g_ca_bin")

plt.imshow(ca_result[:,:,0])
plt.show()