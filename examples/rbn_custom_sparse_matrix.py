""" Simple animation of Echo State Network with custom sparse connection matrix """

import tensorflow as tf
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.custom as conn_custom
import evodynamic.connection as connection
import evodynamic.cells.activation as act
import evodynamic.cells as cells
import networkx as nx
#import time

width = 16

exp = experiment.Experiment()


g_esn = exp.add_cells(name="g_esn", g_cells=cells.Cells(width))
g_esn_real = g_esn.add_real_state(state_name='g_esn_bin')

indices = [[0,0], [0,5], [0,12],\
           [1,6], [1,7], [1,12],\
           [2,5], [2,12], [2,13],\
           [3,0], [3,15], [3,8],\
           [4,11], [4,10], [4,13],\
           [5,13], [5,8], [5,12],\
           [6,9], [6,1], [6,6],\
           [7,8], [7,5], [7,11],\
           [8,14], [8,10], [8,2],\
           [9,14], [9,2], [9,13],\
           [10,6], [10,11], [10,10],\
           [11,7], [11,6], [11,13],\
           [12,11], [12,12], [12,5],\
           [13,10], [13,11], [13,13],\
           [14,3], [14,4], [14,1],\
           [15,0], [15,3], [15,4]]

values = [1,2,4]*width
dense_shape = [width, width]

g_esn_real_conn = conn_custom.create_custom_sparse_matrix('g_ca_bin_conn',
                                                          indices,
                                                          values,
                                                          dense_shape)


exp.add_connection("g_esn_conn",
                   connection.WeightedConnection(g_esn_real,
                                                 g_esn_real,act.sigmoid,
                                                 g_esn_real_conn))

exp.initialize_cells()

weight_matrix = exp.session.run(exp.get_connection("g_esn_conn").w)

G = nx.DiGraph()
G.add_edges_from(weight_matrix[0])

pos = nx.spring_layout(G)
#min_x_val = min([p[0] for p in pos.values()])
#pos_new = {k: (pos[k][0]+min_x_val-1, pos[k][1]) if k<input_size else pos[k] for k in pos.keys()}

# Animation
import matplotlib.pyplot as plt
#import matplotlib.animation as animation

fig, ax = plt.subplots()

plt.title('Step: 0')
current_state = exp.get_group_cells_state("g_esn", "g_esn_bin")

node_color = ["black" if np.random.random()>0.5 else "gray" for node in G]

nx.draw(G.reverse(), node_color = node_color, pos=pos, cmap="gray",
        connectionstyle="arc3, rad=0.1", labels={i:i for i in range(width)}, font_color="w")

#nx.draw(G.reverse(), node_color = node_color, pos=pos, cmap="gray",
#        labels={i:i for i in range(width)}, font_color="w")

#idx_anim = 0
#
#def updatefig(*args):
#  global idx_anim
#
#  ax.clear()
#
#  exp.run_step(feed_dict={input_esn: 1-2*np.random.randint(2, size=(input_size,))})
#
#  current_state = exp.get_group_cells_state("g_esn", "g_esn_bin")
#
#  node_color = [round(current_state[node],2) for node in G]
#  nx.draw(G.reverse(), node_color = node_color, pos=pos_new, cmap=plt.cm.jet,
#          connectionstyle="arc3, rad=0.1")
#
#  plt.title('Step: '+str(idx_anim))
#  idx_anim += 1
#
#
#ani = animation.FuncAnimation(fig, updatefig, frames=30, interval=2000, blit=False)
#
plt.show()
#
## Set up formatting for the movie files
##Writer = animation.writers['ffmpeg']
##writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
##ani.save('results/simple_esn_'+time.strftime("%Y%m%d-%H%M%S")+'.mp4', writer=writer)
#
#
#plt.connect('close_event', exp.close())