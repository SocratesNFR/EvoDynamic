""" Cellular automata 1D - save image"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import numpy as np
from PIL import Image

width = 100
timesteps = 200

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
                             activation_func=act.rule_binary_ca_1d_width3_func,\
                             fargs=(30,))

g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
                             activation_func=act.rule_binary_ca_1d_width3_func,\
                             fargs=(10,))

exp.add_monitor("g_ca", "g_ca_bin")

exp.initialize_cells()

exp.run(timesteps=timesteps)
ca_result = np.invert(exp.get_monitor("g_ca", "g_ca_bin").astype(np.bool)).astype(np.uint8)*255

img = Image.fromarray(ca_result).resize((5*width,5*timesteps), Image.NEAREST)
img.save("ca_1d_randominit_rule30+10.png")