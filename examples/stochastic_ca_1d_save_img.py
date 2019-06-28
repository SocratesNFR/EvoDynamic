""" Cellular automata 1D - Self-organized criticality"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np
import time
from PIL import Image

width = 100
timesteps = 1000

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx,
                                           is_wrapped_ca=True)

#g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
#                            activation_func=act.rule_binary_ca_1d_width3_func,\
#                             fargs=(110,))

#g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
#                             activation_func=act.rule_binary_ca_1d_width3_func,\
#                             fargs=(10,))

fargs_list = [([0.2992531667235731, 0.055125601105234484, 0.32609747643731385,\
                0.0, 0.11885547205711025, 0.2375522459228787, 1.0, 0.4898738531053132],)]

exp.add_connection("g_ca_conn",
                     connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                   act.rule_binary_sca_1d_width3_func,
                                                   g_ca_bin_conn, fargs_list=fargs_list))

exp.add_monitor("g_ca", "g_ca_bin", timesteps)

exp.initialize_cells()

start = time.time()

exp.run(timesteps=timesteps)

print("Execution time:", time.time()-start)

exp.close()

ca_result = np.invert(exp.get_monitor("g_ca", "g_ca_bin").astype(np.bool)).astype(np.uint8)*255

img = Image.fromarray(ca_result).resize((5*width,5*timesteps), Image.NEAREST)
img.save("evolved_stochastic_ca_v2.png")
