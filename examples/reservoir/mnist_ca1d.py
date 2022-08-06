""" Cellular automata 1D - Reservoir for MNIST digit classification """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection as connection
import evodynamic.connection.random as randon_conn
import evodynamic.cells.activation as act
import evodynamic.utils as utils
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

epochs = 1
batch_size = 100
num_batches =  int(np.ceil(x_train_num_images / batch_size))
width = 28*28
timesteps = 28*28
input_size = 1
output_layer_size = 10
image_num_pixels = x_train_image_shape[0] * x_train_image_shape[1]

exp = experiment.Experiment(input_start=0,input_delay=0,training_start=timesteps,
                            training_delay=timesteps,reset_cells_after_train=False,
                            batch_size=batch_size)

input_ca = exp.add_input(tf.float64, [input_size], "input_ca")
desired_output = exp.add_desired_output(tf.float64, [output_layer_size], "desired_output")

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

output_layer =  exp.add_group_cells(name="output_layer", amount=output_layer_size)
output_layer_real_state = output_layer.add_real_state(state_name='output_layer_real_state')

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

writer = tf.summary.FileWriter("output_ca1d", exp.session.graph)

for epoch in range(epochs):
  print("Epoch:", epoch)
  shuffled_indices = np.random.permutation(x_train_num_images)
  batch_indices = np.split(shuffled_indices,\
                           np.arange(batch_size,x_train_num_images,batch_size))
  for step, batch_idx in enumerate(batch_indices):
    start_time = time.time()
    for pixel_idx in range(timesteps):

      input_ca_batch = np.expand_dims(x_train[pixel_idx,batch_idx],0)

      desired_output_batch = y_train[:,batch_idx]

      feed_dict = {input_ca: input_ca_batch, desired_output: desired_output_batch}
      exp.run_step(feed_dict=feed_dict)

    prediction_batch = exp.get_group_cells_state("output_layer", "output_layer_real_state")
    accuracy_batch = np.sum(np.argmax(prediction_batch, axis=0) == np.argmax(desired_output_batch, axis=0)) / batch_size
    utils.progressbar_loss_accu_time(step+1, num_batches-1, exp.training_loss, accuracy_batch, time.time()-start_time)

writer.close()