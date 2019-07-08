""" Cellular automata 1D - Self-organized criticality"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np
import time
from PIL import Image

width = 200
timesteps = 400

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
#
#fargs_list = [([0.8653582694285038, 0.07574324505979713, 0.0,\
#                0.6701419392338681, 0.02243120000370638, 0.46365825379340864,\
#                0.7473194740998363, 1.0],)]


#[0.8844229523381606, 0.16105272036927262, 0.531159153794522, 0.10896772965937962, 0.03419006259328916, 0.8514302121431078, 0.993618976607495, 0.9986691476433076]

#fargs_list = [([0.8844229523381606, 0.16105272036927262, 0.531159153794522,\
#                0.10896772965937962, 0.03419006259328916, 0.8514302121431078,\
#                0.993618976607495, 0.9986691476433076],)]

#fargs_list = [([0.2992531667235731, 0.055125601105234484, 0.32609747643731385,\
#                0.0, 0.11885547205711025, 0.2375522459228787, 1.0, 0.4898738531053132],)]


#TESTE
#[0.9434366987096601, 0.0, 0.2664803614620113, 1.0, 0.0, 0.5734834314810755, 0.004014194229735535, 0.6023044067207612]
#fargs_list = [([0.9434366987096601, 0.0, 0.2664803614620113, 1.0, 0.0,\
#                0.5734834314810755, 0.004014194229735535, 0.6023044067207612],)]

#Test 2
#[0.4934885953923782, 0.4661613668731218, 1.0, 0.3138882790836618, 0.0, 0.007657978203402652, 0.6245438662473278, 0.9949950700207485]
#fargs_list = [([0.4934885953923782, 0.4661613668731218, 1.0, 0.3138882790836618,\
#                0.0, 0.007657978203402652, 0.6245438662473278, 0.9949950700207485],)]


fargs_list = [([0.4073302786551547, 0.07588345236524413, 0.15676889915093514,\
                0.08356471216999127, 0.00879616388063742, 0.403847608690661,\
                0.6751633501344202, 0.8846790551722384],)]

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
img.save("evolved_stochastic_ca_v3_1.png")
