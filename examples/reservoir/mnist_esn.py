""" Echo State Network - Reservoir for MNIST digit classification """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.random as conn_random
import evodynamic.connection as connection
import evodynamic.connection.random as randon_conn
import evodynamic.cells.activation as act
import evodynamic.utils as utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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
width = 2000
timesteps = 28*28
input_size = 28*28
output_layer_size = 10
image_num_pixels = x_train_image_shape[0] * x_train_image_shape[1]

exp = experiment.Experiment(input_start=0,training_start=5,
                            training_delay=5, input_delay_until_train=True,
                            reset_cells_after_train=True,
                            batch_size=batch_size)


input_esn = exp.add_input(tf.float64, [input_size], "input_esn")
input_esn_conn = conn_random.create_truncated_normal_connection("input_esn_conn", input_size, width, stddev=1.0)
desired_output = exp.add_input(tf.float64, [output_layer_size], "desired_output")

g_esn = exp.add_group_cells(name="g_esn", amount=width)
g_esn_real = g_esn.add_real_state(state_name='g_esn_real')
g_esn_real_conn = conn_random.create_gaussian_matrix('g_esn_real_conn',width, sparsity=0.0, is_sparse=False)

exp.add_connection("input_conn",
                   connection.WeightedConnection(input_esn,
                                                 g_esn_real,act.stochastic_sigmoid,
                                                 input_esn_conn))

exp.add_connection("g_esn_conn",
                   connection.WeightedConnection(g_esn_real,
                                                 g_esn_real,act.stochastic_sigmoid,
                                                 g_esn_real_conn))

output_layer =  exp.add_group_cells(name="output_layer", amount=output_layer_size)
output_layer_real_state = output_layer.add_real_state(state_name='output_layer_real_state', stddev=0)

esn_output_conn = randon_conn.create_xavier_connection("esn_output_conn", width, output_layer_size)
exp.add_trainable_connection("output_conn",
                             connection.WeightedConnection(g_esn_real,
                                                           output_layer_real_state,
                                                           act.sigmoid,
                                                           esn_output_conn))

c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=exp.trainable_connections["output_conn"].output,
    labels=desired_output,
    axis=0))

#c_loss = tf.losses.mean_squared_error(labels=desired_output,\
#                                      predictions=exp.trainable_connections["output_conn"].output)

exp.set_training(c_loss,0.003)

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
        im = plt.imshow(weight.reshape((40,50)), cmap="seismic_r", vmin=-max_abs_val, vmax=max_abs_val)

    # Adding colorbar
    # https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, ticks=[-max_abs_val, 0, max_abs_val])

    return fig

output_folder = "weights_esn_22Jun2020"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for epoch in range(epochs):
  print("Epoch:", epoch)
  shuffled_indices = np.random.permutation(x_train_num_images)
  batch_indices = np.split(shuffled_indices,\
                           np.arange(batch_size,x_train_num_images,batch_size))
  for step, batch_idx in enumerate(batch_indices):
    while True:
      input_esn_batch = x_train[:,batch_idx]
      desired_output_batch = y_train[:,batch_idx]
      print(exp.step_counter, exp.is_input_step(), exp.is_training_step())
      feed_dict = {input_esn: input_esn_batch, desired_output: desired_output_batch}
      exp.run_step(feed_dict=feed_dict)
      if exp.is_training_step():
        break


    prediction_batch = exp.get_group_cells_state("output_layer", "output_layer_real_state")
    accuracy_batch = np.sum(np.argmax(prediction_batch, axis=0) == np.argmax(desired_output_batch, axis=0)) / batch_size
    weight = exp.session.run(exp.connections["output_conn"].w)
    fig = plot_first_hidden(np.transpose(weight))
    plt.savefig(output_folder+"\hidden_"+str(exp.step_counter).zfill(6)+'.png', bbox_inches='tight')
    plt.close(fig)
    utils.progressbar_loss_accu(step+1, num_batches-1, exp.training_loss, accuracy_batch)