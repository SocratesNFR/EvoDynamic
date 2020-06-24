""" Cellular automata 1D - Reservoir for MNIST digit classification """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection as connection
import evodynamic.connection.random as randon_conn
import evodynamic.cells.activation as act

mnist = tf.keras.datasets.mnist

(x_train, y_train), _ = mnist.load_data()

x_train_num_images = x_train.shape[0]
x_train_image_shape = x_train.shape[1:3]

x_train= ((x_train / 255.0) > 0.5).astype(np.int8)
x_train = x_train.reshape(x_train.shape[0],-1)
x_train = np.transpose(x_train)

y_train_one_hot = np.zeros((y_train.max()+1, y_train.size))
y_train_one_hot[y_train,np.arange(y_train.size)] = 1
y_train = y_train_one_hot

width = 28*28
timesteps = 28*28
input_size = 1
output_layer_size = 10
image_num_pixels = x_train_image_shape[0] * x_train_image_shape[1]
height_fig = timesteps

exp = experiment.Experiment(input_start=0,input_delay=0,training_start=timesteps-1,
                            training_delay=timesteps-1,reset_cells_after_train=True)

input_ca = exp.add_input(tf.float64, [input_size], "input_ca")
desired_output = exp.add_input(tf.float64, [output_layer_size], "desired_output")

g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init="zeros")
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

fargs_list = [(a,) for a in [170]]

exp.add_connection("input_conn", connection.IndexConnection(input_ca,g_ca_bin,
                                                            [width-1]))

exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))

output_layer = exp.add_group_cells(name="output_layer", amount=output_layer_size)
output_layer_real_state = output_layer.add_real_state(state_name='output_layer_real_state', stddev=0)

ca_output_conn = randon_conn.create_xavier_connection("ca_output_conn", width, output_layer_size)
exp.add_trainable_connection("output_conn",
                             connection.WeightedConnection(g_ca_bin,
                                                           output_layer_real_state,
                                                           act.sigmoid,
                                                           ca_output_conn))

c_loss = tf.losses.mean_squared_error(labels=desired_output, predictions=exp.trainable_connections["output_conn"].output)

exp.set_training(c_loss,0.003)

exp.initialize_cells()

# Animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

idx_anim = 0

im_mnist = x_train[:,0].reshape(x_train_image_shape)
im_ca = np.zeros((height_fig,width))
im_ca[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:,0]
im_output = exp.get_group_cells_state("output_layer", "output_layer_real_state")[:,0].reshape((-1,1))

im1 = ax1.imshow(im_mnist, vmin=0, vmax=1, animated=True)
im2 = ax2.imshow(im_ca, vmin=0, vmax=1, animated=True)
im3 = ax3.imshow(im_output, vmin=0, vmax=1, animated=True)

ax1.title.set_text("Input image")
ax2.title.set_text("CA")
ax3.title.set_text("Trained output")

def updatefig(*args):
    global idx_anim, im_mnist, im_ca, im_output,im1,im2,im3
    idx_anim += 1

    input_ca_np = x_train[idx_anim, idx_anim // timesteps].reshape((-1,1))\
        if idx_anim < image_num_pixels else\
        np.zeros_like(x_train[0,idx_anim // timesteps].reshape((-1,1)))
    desired_output_np = y_train[:,idx_anim // timesteps].reshape((-1,1))
    feed_dict={input_ca: input_ca_np, desired_output: desired_output_np}
    exp.run_step(feed_dict=feed_dict)
    if idx_anim % timesteps == 0:
      im_mnist = x_train[:,idx_anim//timesteps].reshape(x_train_image_shape)
      im_ca = np.zeros((height_fig,width))

    im_ca[idx_anim % timesteps] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:,0]
    im_output = exp.get_group_cells_state("output_layer", "output_layer_real_state")[:,0].reshape((-1,1))

    im1.set_array(im_mnist)
    im2.set_array(im_ca)
    im3.set_array(im_output)

    return im1,im2,im3

ani = animation.FuncAnimation(fig, updatefig, interval=1, blit=False)

plt.show()

plt.connect('close_event', exp.close())
