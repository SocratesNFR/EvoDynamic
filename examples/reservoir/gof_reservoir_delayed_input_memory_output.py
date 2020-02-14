""" Game of life """

import tensorflow as tf
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection as connection
import evodynamic.connection.random as randon_conn
import evodynamic.cells.activation as act
import evodynamic.cells as cells

width = 50
height = 40
input_size = 5*width

exp = experiment.Experiment()

input_ca = exp.add_input(tf.float64, [input_size], "input_ca")
desired_output = exp.add_input(tf.float64, [height], "desired_output")

g_ca = exp.add_cells(name="g_ca", g_cells=cells.Cells(width*height, virtual_shape=(width,height)))
neighbors, center_idx = ca.create_count_neighbors_ca2d(3,3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca2d('g_ca_bin_conn',width,height,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

g_ca_selected = exp.add_cells(name="g_ca_selected", g_cells=cells.Cells(width*height//2))
g_ca_selected_bin = g_ca_selected.add_binary_state(state_name='g_ca_selected_bin', init="zeros")


output_layer = exp.add_cells(name="output_layer", g_cells=cells.Cells(height))
output_layer_real_state = output_layer.add_real_state(state_name='output_layer_real_state', stddev=0)

ca_output_conn = randon_conn.create_xavier_connection("ca_output_conn", width*height//2, height)

exp.add_connection("input_conn", connection.IndexConnection(input_ca,g_ca_bin,
                                                            np.arange(input_size)))
exp.add_connection("g_ca_conn", connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                              act.game_of_life_func,g_ca_bin_conn))

exp.add_connection("g_ca_selected_conn", connection.GatherIndexConnection(g_ca_bin,g_ca_selected_bin,
                                                                          np.arange(input_size, input_size+(width*height//2))))

exp.add_trainable_connection("output_conn",
                             connection.WeightedConnection(g_ca_selected_bin,
                                                           output_layer_real_state,
                                                           act.sigmoid,
                                                           ca_output_conn))

c_loss = tf.losses.mean_squared_error(labels=desired_output, predictions=exp.trainable_connections["output_conn"].output)

exp.set_training(c_loss,0.1)

exp.initialize_cells()

# Animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

idx_anim = 0
arr = np.hstack((exp.get_group_cells_state("g_ca", "g_ca_bin").reshape((height,width)),
                 exp.get_group_cells_state("output_layer", "output_layer_real_state").reshape((height,1))))
im = plt.imshow(arr, animated=True)

plt.title('Step: 0')

def updatefig(*args):
    global idx_anim
    exp.run_step(feed_dict={input_ca: np.random.randint(2, size=(input_size,)), desired_output: np.ones((height,))})
    arr = np.hstack((exp.get_group_cells_state("g_ca", "g_ca_bin").reshape((height,width)),
                 exp.get_group_cells_state("output_layer", "output_layer_real_state").reshape((height,1))))
    im.set_array(arr)

    plt.title('Step: '+str(idx_anim))
    idx_anim += 1
    return im,# ttl

ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=False)

plt.show()
plt.connect('close_event', exp.close())