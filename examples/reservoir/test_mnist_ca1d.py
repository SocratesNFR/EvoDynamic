"""
Testing features and method for
Cellular automata 1D - Reservoir for MNIST digit classification
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection as connection
import evodynamic.connection.random as randon_conn
import evodynamic.cells.activation as act
import evodynamic.utils as utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_num_images = x_train.shape[0]
x_train_image_shape = x_train.shape[1:3]

x_train = ((x_train / 255.0) > 0.5).astype(np.int8)
x_train = x_train.reshape(x_train.shape[0],-1)
x_train = np.transpose(x_train)

x_test = ((x_test / 255.0) > 0.5).astype(np.int8)
x_test = x_test.reshape(x_test.shape[0],-1)
x_test = np.transpose(x_test)

y_train_one_hot = np.zeros((y_train.max()+1, y_train.size))
y_train_one_hot[y_train,np.arange(y_train.size)] = 1
y_train = y_train_one_hot

y_test_one_hot = np.zeros((y_test.max()+1, y_test.size))
y_test_one_hot[y_test,np.arange(y_test.size)] = 1
y_test = y_test_one_hot

epochs = 10
batch_size = 100
num_batches =  int(np.ceil(x_train_num_images / batch_size))
width = 28*28
timesteps = 0
input_size = 28*28
output_layer_size = 10
image_num_pixels = x_train_image_shape[0] * x_train_image_shape[1]

exp = experiment.Experiment(input_start=0,input_delay=0,training_start=0,
                            training_delay=0,reset_cells_after_train=False,
                            batch_size=batch_size)

input_ca = exp.add_input(tf.float64, [input_size], "input_ca")
desired_output = exp.add_desired_output(tf.float64, [output_layer_size], "desired_output")

g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init="zeros")
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

fargs_list = [(a,) for a in [51]]


exp.add_connection("input_conn", connection.IndexConnection(input_ca,g_ca_bin,
                                                            np.arange(width)))

exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))

output_layer =  exp.add_group_cells(name="output_layer", amount=output_layer_size)
output_layer_real_state = output_layer.add_real_state(state_name='output_layer_real_state', stddev=0)

ca_output_conn = randon_conn.create_xavier_connection("ca_output_conn", width, output_layer_size)
exp.add_trainable_connection("output_conn",
                             connection.WeightedConnection(g_ca_bin,
                                                           output_layer_real_state,
                                                           act.sigmoid,
                                                           ca_output_conn))

c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=exp.trainable_connections["output_conn"].output,
    labels=desired_output,
    axis=0))

exp.set_training(c_loss,0.03)

exp.initialize_cells()

def plot_first_hidden(weights):
    max_abs_val = max(abs(np.max(weights)), abs(np.min(weights)))
    fig = plt.figure(figsize=(5, 2))
    gs = gridspec.GridSpec(2, 5)
    gs.update(wspace=0.1, hspace=0.1)

    for i, weight in enumerate(np.transpose(weights)):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        im = plt.imshow(weight.reshape((28,28)), cmap="seismic_r", vmin=-max_abs_val, vmax=max_abs_val)

    # Adding colorbar
    # https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, ticks=[-max_abs_val, 0, max_abs_val])

    return fig

output_folder = "test_mnist_ca1d_"+time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for epoch in range(epochs):
  print("Epoch:", epoch)
  shuffled_indices = np.random.permutation(x_train_num_images)
  batch_indices = np.split(shuffled_indices,\
                           np.arange(batch_size,x_train_num_images,batch_size))
  for step, batch_idx in enumerate(batch_indices):


    input_ca_batch = 1. - x_train[:,batch_idx]
    desired_output_batch = y_train[:,batch_idx]

    feed_dict = {input_ca: input_ca_batch, desired_output: desired_output_batch}
    exp.run_step(feed_dict=feed_dict)

    prediction_batch = exp.get_group_cells_state("output_layer", "output_layer_real_state")
    accuracy_batch = np.sum(np.argmax(prediction_batch, axis=0) == np.argmax(desired_output_batch, axis=0)) / batch_size

    weight = exp.session.run(exp.connections["output_conn"].w)

    fig = plot_first_hidden(np.transpose(weight))
    plt.savefig(output_folder+"\hidden_"+str(exp.step_counter).zfill(6)+'.png', bbox_inches='tight')
    plt.close(fig)

    res_ca = exp.get_group_cells_state("g_ca", "g_ca_bin")[:,0]
    fig = plt.figure()
    plt.imsave(output_folder+"\memory_"+str(exp.step_counter).zfill(6)+'.png', res_ca.reshape((28,28)))
    plt.close(fig)

    utils.progressbar_loss_accu(step+1, num_batches-1, exp.training_loss, accuracy_batch)