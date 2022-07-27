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
input_scaling = 0.6
input_sparsity = 0.9
leaky_rate = 0.5

exp = experiment.Experiment()

input_esn = exp.add_input(tf.float64, [input_size], "input_esn")
g_input_real_conn = conn_random.create_gaussian_connection('g_input_real_conn',
                                                          input_size, width,
                                                          scale=input_scaling,
                                                          sparsity=input_sparsity,
                                                          is_sparse=True)

g_esn = exp.add_group_cells(name="g_esn", amount=width)
g_esn_real = g_esn.add_real_state(state_name='g_esn_real')
g_esn_real_conn = conn_random.create_gaussian_matrix('g_esn_real_conn',width,
                                                     spectral_radius=1.3,
                                                     sparsity=0.95, is_sparse=True)
# g_esn_real_bias = conn_random.create_gaussian_connection('g_esn_real_bias',
#                                                          1, width,
#                                                          scale=1.0,
#                                                          is_sparse=False)

exp.add_connection("input_conn",
                   connection.WeightedConnection(input_esn,
                                                 g_esn_real,act.tanh,
                                                 g_input_real_conn))

# exp.add_connection("g_esn_conn",
#                     connection.WeightedConnection(g_esn_real,
#                                                   g_esn_real,act.leaky_sigmoid,
#                                                   g_esn_real_conn,
#                                                   fargs_list=[(leaky_rate,)]))

exp.add_connection("g_esn_conn",
                    connection.WeightedConnection(g_esn_real,
                                                  g_esn_real,act.leaky_tanh,
                                                  g_esn_real_conn,
                                                  fargs_list=[(leaky_rate,)]))

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

nx.draw(G.reverse(), node_color = node_color, pos=pos_new, cmap=plt.cm.coolwarm,
        vmin=-1, vmax=1,
        connectionstyle="arc3, rad=0.1")

idx_anim = 0

def updatefig(*args):
  global idx_anim

  ax.clear()
  input_esn_arr = np.random.randint(2, size=(input_size,1)) if idx_anim < 6 else np.zeros((input_size,1))

  exp.run_step(feed_dict={input_esn: input_esn_arr})

  current_state = exp.get_group_cells_state("g_esn", "g_esn_real")[:,0]

  node_color = [round(current_state[node],2) for node in G]
  nx.draw(G.reverse(), node_color = node_color, pos=pos_new, cmap=plt.cm.coolwarm,
          vmin=-1, vmax=1,
          connectionstyle="arc3, rad=0.1")

  plt.title('Step: '+str(idx_anim))
  idx_anim += 1


ani = animation.FuncAnimation(fig, updatefig, frames=30, interval=2000, blit=False)

plt.show()

plt.connect('close_event', exp.close())