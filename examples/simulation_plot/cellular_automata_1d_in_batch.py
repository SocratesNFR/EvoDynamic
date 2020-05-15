""" Cellular automata 1D in batch - simulation plot """

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import matplotlib.pyplot as plt

width = 100
timesteps = 400
batch_size = 4

exp = experiment.Experiment(batch_size=batch_size)
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

fargs_list = [(a,) for a in [110]]

exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))

exp.add_monitor("g_ca", "g_ca_bin")

exp.initialize_cells()

exp.run(timesteps=timesteps)

ca_result = exp.get_monitor("g_ca", "g_ca_bin")

fig, axs = plt.subplots(1, batch_size)
for i in range(batch_size):
  axs[i].imshow(ca_result[:,i,:])
plt.show()