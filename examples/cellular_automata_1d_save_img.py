""" Cellular automata 1D - save image"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np
from PIL import Image

width = 100
timesteps = 200

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

#fargs_list = [(a,) for a in [54, 52, 144, 127, 30, 17, 159, 81, 75, 226, 222,\
#              16, 191, 206, 14, 133, 237, 191, 204, 92, 98, 8, 55, 202, 169,\
#              243, 221, 80, 102, 154, 186, 125, 2, 172, 237, 242, 184, 140, 208,\
#              34, 248]]

fargs_list = [(a,) for a in [235, 36]]


exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))

exp.add_monitor("g_ca", "g_ca_bin")

exp.initialize_cells()

exp.run(timesteps=timesteps)
ca_result = np.invert(exp.get_monitor("g_ca", "g_ca_bin").astype(np.bool)).astype(np.uint8)*255

img = Image.fromarray(ca_result).resize((5*width,5*timesteps), Image.NEAREST)
img.save("teste_ca1d.png")