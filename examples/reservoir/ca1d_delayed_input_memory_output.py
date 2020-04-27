""" Cellular automata 1D - Reservoir """

import tensorflow as tf
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection as connection
import evodynamic.connection.random as randon_conn
import evodynamic.cells.activation as act
import evodynamic.cells as cells

width = 100
height_fig = 200
input_size = width // 2
output_layer_size = width
memory_size = 5

exp = experiment.Experiment(input_start=5,input_delay=14,training_start=6,training_delay=0)

input_ca = exp.add_input(tf.float64, [input_size], "input_ca")
desired_output = exp.add_input(tf.float64, [output_layer_size], "desired_output")

g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

fargs_list = [(a,) for a in [110]]

exp.add_connection("input_conn", connection.IndexConnection(input_ca,g_ca_bin,
                                                            np.arange(input_size)))

exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))

g_ca_memory = exp.add_state_memory(g_ca_bin,memory_size)

output_layer = exp.add_cells(name="output_layer", g_cells=cells.Cells(output_layer_size))
output_layer_real_state = output_layer.add_real_state(state_name='output_layer_real_state', stddev=0)

ca_output_conn = randon_conn.create_xavier_connection("ca_output_conn", width*memory_size, output_layer_size)
exp.add_trainable_connection("output_conn",
                             connection.WeightedConnection(g_ca_memory,
                                                           output_layer_real_state,
                                                           act.sigmoid,
                                                           ca_output_conn))

c_loss = tf.losses.mean_squared_error(labels=desired_output, predictions=exp.trainable_connections["output_conn"].output)

exp.set_training(c_loss,0.1)

exp.initialize_cells()

# Animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

idx_anim = 0

im_ca = np.zeros((height_fig,width))
im_ca[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")
im_memory = exp.memories[g_ca_bin].get_state_memory().reshape((memory_size, width))
im_output = exp.get_group_cells_state("output_layer", "output_layer_real_state").reshape((-1,1))

im1 = ax1.imshow(im_ca, vmin=0, vmax=1, animated=True)
im2 = ax2.imshow(im_memory, vmin=0, vmax=1, animated=True)
im3 = ax3.imshow(im_output, vmin=0, vmax=1, animated=True)

ax1.title.set_text("CA")
ax2.title.set_text("Memory")
ax3.title.set_text("Trained output")

def updatefig(*args):
    global idx_anim, im_ca, im_memory, im_output,im1,im2,im3
    idx_anim += 1

    input_ca_np = np.zeros((input_size,)) if ((idx_anim // 10) % 2 == 0) else np.ones((input_size,))
    desired_output_np = np.zeros((output_layer_size,)) if (idx_anim//10) % 2 == 2 else np.ones((output_layer_size,))
    feed_dict={input_ca: input_ca_np, desired_output: desired_output_np}
    exp.run_step(feed_dict=feed_dict)

    if idx_anim < height_fig:
      im_ca[idx_anim] = exp.get_group_cells_state("g_ca", "g_ca_bin")
    else:
      im_ca = np.vstack((im_ca[1:], exp.get_group_cells_state("g_ca", "g_ca_bin")))

    im_memory = exp.memories[g_ca_bin].get_state_memory().reshape((memory_size, width))
    im_output = exp.get_group_cells_state("output_layer", "output_layer_real_state").reshape((-1,1))

    im1.set_array(im_ca)
    im2.set_array(im_memory)
    im3.set_array(im_output)

    return im1,im2,im3

ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=False)

plt.show()

plt.connect('close_event', exp.close())
