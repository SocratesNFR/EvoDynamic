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

exp = experiment.Experiment()

input_esn = exp.add_input(tf.float64, [input_size], "input_esn")

g_esn = exp.add_group_cells(name="g_esn", amount=width)
g_esn_real = g_esn.add_real_state(state_name='g_esn_real')
g_esn_real_conn = conn_random.create_gaussian_matrix('g_esn_real_conn',width, sparsity=0.95, is_sparse=True)

exp.add_connection("input_conn", connection.IndexConnection(input_esn,g_esn_real,np.arange(input_size)))

exp.add_connection("g_esn_conn",
                   connection.WeightedConnection(g_esn_real,
                                                 g_esn_real,act.stochastic_sigmoid,
                                                 g_esn_real_conn))

exp.initialize_cells()

weight_matrix = exp.session.run(exp.get_connection("g_esn_conn").w)

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
current_state = exp.get_group_cells_state("g_esn", "g_esn_real")[:,0]

node_color = [round(current_state[node],2) for node in G]

nx.draw(G.reverse(), node_color = node_color, pos=pos_new, cmap=plt.cm.jet,
        connectionstyle="arc3, rad=0.1")

idx_anim = 0

def updatefig(*args):
  global idx_anim

  ax.clear()

  input_esn_arr = np.random.randint(2, size=(input_size,1)) if idx_anim < 1 else np.zeros((input_size,1))

  exp.run_step(feed_dict={input_esn: input_esn_arr})

  current_state = exp.get_group_cells_state("g_esn", "g_esn_real")[:,0]

  node_color = [round(current_state[node],2) for node in G]
  nx.draw(G.reverse(), node_color = node_color, pos=pos_new, cmap=plt.cm.jet,
          connectionstyle="arc3, rad=0.1")

  plt.title('Step: '+str(idx_anim))
  idx_anim += 1


ani = animation.FuncAnimation(fig, updatefig, frames=30, interval=2000, blit=False)

plt.show()

plt.connect('close_event', exp.close())