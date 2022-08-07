"""
Testing features and method for
Liquid State Machine - Reservoir for MNIST digit classification

Adapted from:
Schaetti, Nils, Michel Salomon, and Raphaël Couturier.
"Echo state networks-based reservoir computing for mnist handwritten digits
recognition." 2016 IEEE Intl conference on computational science and
engineering (CSE) and IEEE Intl conference on embedded and ubiquitous computing
(EUC) and 15th Intl symposium on distributed computing and applications for
business engineering (DCABES). IEEE, 2016.
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.random as conn_random
import evodynamic.connection as connection
import evodynamic.cells.activation as act
import evodynamic.utils as utils
import time

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_num_images = x_train.shape[0]
x_train_image_shape = x_train.shape[1:3]
x_test_num_images = x_test.shape[0]
x_test_image_shape = x_test.shape[1:3]

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

epochs = 2
batch_size = 100
num_batches =  int(np.ceil(x_train_num_images / batch_size))
num_batches_test =  int(np.ceil(x_test_num_images / batch_size))
width = 100#1200
image_size = 28
input_size = image_size
memory_size = image_size
input_scaling = 0.6
input_sparsity = 0.9
lsm_sparsity = 0.1
output_layer_size = 10
spectral_radius = 1.3
threshold = 1.0
potential_decay = 0.01
image_num_pixels = x_train_image_shape[0] * x_train_image_shape[1]

exp = experiment.Experiment(input_start=0,input_delay=1,training_start=image_size,
                            training_delay=image_size,reset_cells_after_train=True,
                            batch_size=batch_size)


input_lsm = exp.add_input(tf.float64, [input_size], "input_lsm")
desired_output = exp.add_desired_output(tf.float64, [output_layer_size], "desired_output")

g_lsm = exp.add_group_cells(name="g_lsm", amount=width)
g_lsm_mem = g_lsm.add_real_state(state_name='g_lsm_mem')
g_lsm_spike = g_lsm.add_binary_state(state_name='g_lsm_spike', init ='zeros')

g_input_real_conn = conn_random.create_gaussian_connection('g_input_real_conn',
                                                          input_size, width,
                                                          scale=input_scaling,
                                                          sparsity=input_sparsity,
                                                          is_sparse=False)

exp.add_connection("input_conn",
                   connection.WeightedConnection(input_lsm,
                                                 g_lsm_mem,None,
                                                 g_input_real_conn))

g_lsm_real_conn = conn_random.create_gaussian_matrix('g_lsm_real_conn',
                                                      width,
                                                      spectral_radius=spectral_radius,
                                                      sparsity=lsm_sparsity,
                                                      is_sparse=False)

exp.add_connection("g_lsm_conn",
                    connection.WeightedConnection(g_lsm_spike,
                                                  g_lsm_spike,act.integrate_and_fire,
                                                  g_lsm_real_conn,
                                                  fargs_list=[(g_lsm_mem,threshold,potential_decay)]))

output_layer =  exp.add_group_cells(name="output_layer", amount=output_layer_size)
output_layer_real_state = output_layer.add_real_state(state_name='output_layer_real_state')

g_lsm_memory = exp.add_state_memory(g_lsm_spike,memory_size)
lsm_output_conn = conn_random.create_xavier_connection("lsm_output_conn", width*memory_size, output_layer_size)

lsm_output_bias = conn_random.create_xavier_connection("lsm_output_bias", 1, output_layer_size)

exp.add_trainable_connection("output_conn",
                             connection.BiasWeightedConnection(g_lsm_memory,
                                                               output_layer_real_state,
                                                               act.sigmoid,
                                                               lsm_output_conn,
                                                               lsm_output_bias))

c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=exp.trainable_connections["output_conn"].output,
    labels=desired_output,
    axis=0))

exp.set_training(c_loss,0.001)

exp.initialize_cells()

for epoch in range(epochs):
  print("Epoch:", epoch)
  shuffled_indices = np.random.permutation(x_train_num_images)
  batch_indices = np.split(shuffled_indices,\
                           np.arange(batch_size,x_train_num_images,batch_size))
  accuracy_train = []
  for step, batch_idx in enumerate(batch_indices):
    start_time = time.time()
    for i in range(image_size):
      input_lsm_batch = x_train[i*input_size:(i+1)*input_size,batch_idx]
      desired_output_batch = y_train[:,batch_idx]
      feed_dict = {input_lsm: input_lsm_batch, desired_output: desired_output_batch}
      exp.run_step(feed_dict=feed_dict)

    prediction_batch = exp.get_group_cells_state("output_layer", "output_layer_real_state")
    accuracy_batch = np.mean(np.argmax(prediction_batch, axis=0) == np.argmax(desired_output_batch, axis=0))
    accuracy_train.append(accuracy_batch)
    utils.progressbar_loss_accu_time(step+1, num_batches, exp.training_loss, accuracy_batch, time.time()-start_time)
  print("Training average accuracy:", np.mean(accuracy_train))

  print("Testing...")
  # Testing!
  shuffled_indices_test = np.random.permutation(x_test_num_images)
  batch_indices_test = np.split(shuffled_indices_test,\
                           np.arange(batch_size,x_test_num_images,batch_size))
  accuracy_test = []
  for step_test, batch_idx in enumerate(batch_indices_test):
    start_time = time.time()
    for i in range(image_size):
      input_lsm_batch = x_test[i*input_size:(i+1)*input_size,batch_idx]
      desired_output_batch = y_test[:,batch_idx]
      feed_dict = {input_lsm: input_lsm_batch, desired_output: desired_output_batch}
      exp.run_step(feed_dict=feed_dict, testing=True)

    prediction_batch = exp.get_group_cells_state("output_layer", "output_layer_real_state")
    accuracy_batch = np.mean(np.argmax(prediction_batch, axis=0) == np.argmax(desired_output_batch, axis=0))
    accuracy_test.append(accuracy_batch)

    utils.progressbar_loss_accu_time(step_test+1, num_batches_test, exp.training_loss, accuracy_batch, time.time()-start_time)
  print("Test average accuracy:", np.mean(accuracy_test))
