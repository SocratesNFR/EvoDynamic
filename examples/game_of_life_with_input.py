""" Game of life """

import tensorflow as tf
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection as connection
import evodynamic.cells.activation as act
import evodynamic.cells as cells

width = 200
height = 150
input_size = 5*width

exp = experiment.Experiment()

input_ca = exp.add_input(tf.float64, [input_size], "input_ca")

g_ca = exp.add_cells(name="g_ca", g_cells=cells.Cells(width*height, virtual_shape=(width,height)))
neighbors, center_idx = ca.create_count_neighbors_ca2d(3,3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca2d('g_ca_bin_conn',width,height,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

exp.add_connections("input_conn", connection.IndexConnection(input_ca,g_ca_bin,np.arange(input_size)))
exp.add_connections("g_ca_conn", connection.WeightedConnection(g_ca_bin,g_ca_bin,act.game_of_life_func,g_ca_bin_conn))

exp.initialize_cells()

# Animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

idx_anim = 0
im = plt.imshow(exp.get_group_cells_state("g_ca", "g_ca_bin").reshape((height,width)), animated=True)

plt.title('Step: 0')

def updatefig(*args):
    global idx_anim
    exp.run_step(feed_dict={input_ca: np.random.randint(2, size=(input_size,))})
    im.set_array(exp.get_group_cells_state("g_ca", "g_ca_bin").reshape((height,width)))

    plt.title('Step: '+str(idx_anim))
    idx_anim += 1
    return im,# ttl

ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=False)

plt.show()
plt.connect('close_event', exp.close())