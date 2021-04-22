"""
Testing features and method for
Echo State Network - Reservoir for MNIST digit classification
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.random as conn_random
import evodynamic.connection as connection
import evodynamic.connection.custom as conn_custom
import evodynamic.cells.activation as act
import evodynamic.utils as utils

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_num_images = x_train.shape[0]
x_train_image_shape = x_train.shape[1:3]

x_train = ((x_train / 255.0) > 0.5).astype(np.float64)
x_train = x_train.reshape(x_train.shape[0],-1)
x_train = np.transpose(x_train)

x_test = ((x_test / 255.0) > 0.5).astype(np.float64)
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
width = 2*28*28
input_size = 28*28
output_layer_size = 10
image_num_pixels = x_train_image_shape[0] * x_train_image_shape[1]
threshold = 1.0
potential_decay = 0.01

exp = experiment.Experiment(input_start=0,input_delay=0,training_start=0,
                            training_delay=0,reset_cells_after_train=True,
                            batch_size=batch_size)


input_lsm = exp.add_input(tf.float64, [input_size], "input_esn")
desired_output = exp.add_desired_output(tf.float64, [output_layer_size], "desired_output")

g_lsm = exp.add_group_cells(name="g_lsm", amount=width)
g_lsm_mem = g_lsm.add_real_state(state_name='g_lsm_mem')
g_lsm_spike = g_lsm.add_binary_state(state_name='g_lsm_spike', init ='zeros')
g_lsm_conn = conn_random.create_gaussian_matrix('g_lsm_conn',width, sparsity=0.95, is_sparse=True)
# create_uniform_connection(name, from_group_amount, to_group_amount, sparsity=None, is_sparse=False)
g_lsm_input_conn = conn_random.create_uniform_connection('g_lsm_input_conn', input_size, width, sparsity=0.9)



exp.add_connection("input_conn", connection.WeightedConnection(input_lsm,
                                                              g_lsm_spike, act.integrate_and_fire,
                                                              g_lsm_input_conn,
                                                              fargs_list=[(g_lsm_mem,threshold,potential_decay)]))

exp.add_connection("g_lsm_conn",
                   connection.WeightedConnection(g_lsm_spike,
                                                 g_lsm_spike, act.integrate_and_fire,
                                                 g_lsm_conn,
                                                 fargs_list=[(g_lsm_mem,threshold,potential_decay)]))

output_layer =  exp.add_group_cells(name="output_layer", amount=output_layer_size)
output_layer_real_state = output_layer.add_real_state(state_name='output_layer_real_state')

lsm_output_conn = conn_random.create_xavier_connection("lsm_output_conn", width, output_layer_size)
exp.add_trainable_connection("output_conn",
                             connection.WeightedConnection(g_lsm_spike,
                                                           output_layer_real_state,
                                                           act.sigmoid,
                                                           lsm_output_conn))

c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=exp.trainable_connections["output_conn"].output,
    labels=desired_output,
    axis=0))

exp.set_training(c_loss,0.03)

# Monitors are needed because "reset_cells_after_train=True"
exp.add_monitor("output_layer", "output_layer_real_state", timesteps=1)
exp.add_monitor("g_lsm", "g_lsm_spike", timesteps=1)

exp.initialize_cells()

for epoch in range(epochs):
  print("Epoch:", epoch)
  shuffled_indices = np.random.permutation(x_train_num_images)
  batch_indices = np.split(shuffled_indices,\
                           np.arange(batch_size,x_train_num_images,batch_size))
  for step, batch_idx in enumerate(batch_indices):
    input_lsm_batch = x_train[:,batch_idx]
    desired_output_batch = y_train[:,batch_idx]
    feed_dict = {input_lsm: input_lsm_batch, desired_output: desired_output_batch}
    exp.run_step(feed_dict=feed_dict)

    prediction_batch = exp.get_monitor("output_layer", "output_layer_real_state")[0,:,:]
    accuracy_batch = np.sum(np.argmax(prediction_batch, axis=0) == np.argmax(desired_output_batch, axis=0)) / batch_size

    utils.progressbar_loss_accu(step+1, num_batches, exp.training_loss, accuracy_batch)
