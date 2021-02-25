""" Simple animation of Echo State Network """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.random as conn_random
import evodynamic.connection as connection
import evodynamic.cells.activation as act
import networkx as nx

width = 100
input_size = width // 10

a, b, c, d = 0.02, 0.20, -65.0, 8.00
dt = 0.5

exp = experiment.Experiment()

input_lsm = exp.add_input(tf.float64, [input_size], "input_lsm")

g_lsm = exp.add_group_cells(name="g_lsm", amount=width)
g_lsm_mem = g_lsm.add_real_state(state_name='g_lsm_mem', init_full = c)
g_lsm_rec = g_lsm.add_real_state(state_name='g_lsm_rec', init_full = b*c)
g_lsm_spike = g_lsm.add_binary_state(state_name='g_lsm_spike', init ='zeros')
g_lsm_conn = conn_random.create_gaussian_matrix('g_lsm_conn', width, mean=10.0, std=5.0, sparsity=0.95, is_sparse=True)
# create_uniform_connection(name, from_group_amount, to_group_amount, sparsity=None, is_sparse=False)
g_lsm_input_conn = conn_random.create_gaussian_connection('g_lsm_input_conn', input_size, width, mean=10.0, std=5.0, sparsity=0.9)


#izhikevich(potential_change, spike_in, potential, recovery, a, b, c, d)

exp.add_connection("input_conn", connection.WeightedConnection(input_lsm,
                                                              g_lsm_spike, act.izhikevich,
                                                              g_lsm_input_conn,
                                                              fargs_list=[(g_lsm_mem,g_lsm_rec,a, b, c, d, dt)]))

exp.add_connection("g_lsm_conn",
                   connection.WeightedConnection(g_lsm_spike,
                                                 g_lsm_spike, act.izhikevich,
                                                 g_lsm_conn,
                                                 fargs_list=[(g_lsm_mem,g_lsm_rec,a, b, c, d, dt)]))

exp.initialize_cells()

weight_matrix = exp.session.run(exp.get_connection("g_lsm_conn").w)

G = nx.DiGraph()
G.add_edges_from(weight_matrix[0])

pos_fixed = nx.spring_layout(G)

# Animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, axs = plt.subplots(nrows=1, ncols=3)

plt.title('Step: 0')
current_mem = exp.get_group_cells_state("g_lsm", "g_lsm_mem")[:,0]
current_spike = exp.get_group_cells_state("g_lsm", "g_lsm_spike")[:,0]

node_color = [round(current_spike[node],2) for node in G]

nx.draw(G.reverse(), node_color = node_color, pos=pos_fixed, cmap=plt.cm.jet,
        connectionstyle="arc3, rad=0.1", ax=axs[0])

idx_plot_neuron = 50
plot_width = 200
x_values = np.array([0])
mem_values = np.array([round(current_mem[idx_plot_neuron],2)])
spike_values, = np.where(current_spike == 1)
scatter_values = np.full_like(spike_values, 0)
axs_1, = axs[1].plot(x_values, mem_values)
axs_2 = axs[2].scatter(scatter_values, spike_values)

xmin, xmax = np.min(x_values), np.min(x_values)+plot_width
ymin, ymax = -100.0, 100.0
axs[1].set(xlim=(xmin, xmax), ylim=(ymin, ymax))
axs[2].set(xlim=(xmin, xmax), ylim=(-1, width))

idx_anim = 0

def updatefig(*args):
  global idx_anim, x_values, mem_values, spike_values, scatter_values#, axs_1, axs_2

  axs[0].clear()

  exp.run_step(feed_dict={input_lsm: 10*np.random.randint(2, size=(input_size,1))})

  current_mem = exp.get_group_cells_state("g_lsm", "g_lsm_mem")[:,0]
  current_spike = exp.get_group_cells_state("g_lsm", "g_lsm_spike")[:,0]


  current_spike_values = np.where(current_spike == 1)[0]
  current_scatter_values = np.full_like(current_spike_values, idx_anim+1)
  if idx_anim >= plot_width:
    x_values = np.concatenate((x_values[1:], [np.max(x_values)+1]))
    mem_values = np.concatenate((mem_values[1:], [current_mem[idx_plot_neuron]]))
    spike_values =  np.concatenate((spike_values[scatter_values>(idx_anim-plot_width)], current_spike_values))
    scatter_values = np.concatenate((scatter_values[scatter_values>(idx_anim-plot_width)], current_scatter_values))
  else:
    x_values = np.concatenate((x_values, [np.max(x_values)+1]))
    mem_values = np.concatenate((mem_values, [current_mem[idx_plot_neuron]]))
    spike_values =  np.concatenate((spike_values, np.where(current_spike == 1)[0]))
    scatter_values = np.concatenate((scatter_values, current_scatter_values))

  node_color = [round(current_spike[node],2) for node in G]
  nx.draw(G.reverse(), node_color = node_color, pos=pos_fixed, cmap=plt.cm.jet,
          connectionstyle="arc3, rad=0.1", ax=axs[0])

  xmin, xmax = np.min(x_values), np.min(x_values)+plot_width
  axs[1].set(xlim=(xmin, xmax))
  axs[2].set(xlim=(xmin, xmax))

  axs_1.set_data(x_values, mem_values)
  axs_2.set_offsets(np.vstack((scatter_values, spike_values)).transpose())

  plt.title('Step: '+str(idx_anim))
  idx_anim += 1


ani = animation.FuncAnimation(fig, updatefig, frames=30, interval=500, blit=False)

plt.show()

plt.connect('close_event', exp.close())