""" Simple animation of Echo State Network with custom connection matrix"""

import tensorflow as tf
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.custom as conn_custom
import evodynamic.connection as connection
import evodynamic.cells.activation as act
import evodynamic.cells as cells
import networkx as nx

width = 100
input_size = width // 10

exp = experiment.Experiment()

input_esn = exp.add_input(tf.float64, [input_size], "input_esn")

g_esn = exp.add_cells(name="g_esn", g_cells=cells.Cells(width))
g_esn_real = g_esn.add_real_state(state_name='g_esn_bin')

# Generete custom connection matrix
conn_matrix = np.random.normal(loc=0.0, scale=0.4, size=(width, width))
conn_matrix[np.round(conn_matrix) == 0.0] = 0.0


g_esn_real_conn = conn_custom.create_custom_matrix('g_ca_bin_conn',conn_matrix)

exp.add_connection("input_conn", connection.IndexConnection(input_esn,g_esn_real,np.arange(input_size)))

exp.add_connection("g_esn_conn",
                   connection.WeightedConnection(g_esn_real,
                                                 g_esn_real,act.sigmoid,
                                                 g_esn_real_conn))

exp.initialize_cells()

weight_matrix = exp.session.run(exp.get_connection("g_esn_conn").w)

def adjacency2indices_values(adjacency_matrix):
  indices = np.argwhere(adjacency_matrix != 0)
  values = adjacency_matrix[np.where(adjacency_matrix != 0)]

  return indices, values

indices, values = adjacency2indices_values(weight_matrix)

G = nx.DiGraph()
G.add_edges_from(indices)

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
current_state = exp.get_group_cells_state("g_esn", "g_esn_bin")

node_color = [round(current_state[node],2) for node in G]

nx.draw(G.reverse(), node_color = node_color, pos=pos_new, cmap=plt.cm.jet,
        connectionstyle="arc3, rad=0.1")

idx_anim = 0

def updatefig(*args):
  global idx_anim

  ax.clear()

  exp.run_step(feed_dict={input_esn: np.random.randint(2, size=(input_size,))})

  current_state = exp.get_group_cells_state("g_esn", "g_esn_bin")

  node_color = [round(current_state[node],2) for node in G]
  nx.draw(G.reverse(), node_color = node_color, pos=pos_new, cmap=plt.cm.jet,
          connectionstyle="arc3, rad=0.1")

  plt.title('Step: '+str(idx_anim))
  idx_anim += 1


ani = animation.FuncAnimation(fig, updatefig, frames=30, interval=2000, blit=False)

plt.show()

# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#ani.save('results/simple_esn_'+time.strftime("%Y%m%d-%H%M%S")+'.mp4', writer=writer)


plt.connect('close_event', exp.close())