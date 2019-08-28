""" Simple animation of random Boolean Network (rule 110) """

import tensorflow as tf
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.random_boolean_net as rbn
import evodynamic.connection as connection
import evodynamic.cells.activation as act
import evodynamic.cells as cells
import networkx as nx

width = 100
input_size = width // 10

exp = experiment.Experiment()

input_rbn = exp.add_input(tf.float64, [input_size], "input_rbn")

g_rbn = exp.add_cells(name="g_rbn", g_cells=cells.Cells(width))
g_rbn_bin = g_rbn.add_binary_state(state_name='g_rbn_bin')
g_rbn_bin_conn = rbn.create_conn_matrix('g_ca_bin_conn',width)

exp.add_connection("input_conn", connection.IndexConnection(input_rbn,g_rbn_bin,np.arange(input_size)))

fargs_list = [(a,) for a in [110]]
exp.add_connection("g_rbn_conn",
                   connection.WeightedConnection(g_rbn_bin,
                                                 g_rbn_bin,act.rule_binary_ca_1d_width3_func,
                                                 g_rbn_bin_conn,fargs_list=fargs_list))

exp.initialize_cells()

weight_matrix = exp.session.run(exp.get_connection("g_rbn_conn").w)

G = nx.DiGraph()
G.add_edges_from(weight_matrix[0])

pos_dict = {}
for i in range(width):
  if i < input_size:
    pos_dict[i] = (0,i)

pos = nx.spring_layout(G,pos=pos_dict, fixed=pos_dict.keys())
min_x_val = min([p[0] for p in pos.values()])
pos_new = {k: (pos[k][0]+min_x_val-1, pos[k][1]) if k<input_size else pos[k] for k in pos.keys()}

# Animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

plt.title('Step: 0')
current_state = exp.get_group_cells_state("g_rbn", "g_rbn_bin")

node_color = ["black" if current_state[node]==0 else "gray" for node in G]

nx.draw(G.reverse(), node_color = node_color, pos=pos_new,
        connectionstyle="arc3, rad=0.1")

idx_anim = 0

def updatefig(*args):
  global idx_anim

  ax.clear()

  exp.run_step(feed_dict={input_rbn: np.random.randint(2, size=(input_size,))})

  current_state = exp.get_group_cells_state("g_rbn", "g_rbn_bin")

  node_color = ["black" if current_state[node]==0 else "gray" for node in G]
  
  nx.draw(G.reverse(), node_color = node_color, pos=pos_new,
          connectionstyle="arc3, rad=0.1")

  plt.title('Step: '+str(idx_anim))
  idx_anim += 1


ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=False)

plt.show()
plt.connect('close_event', exp.close())