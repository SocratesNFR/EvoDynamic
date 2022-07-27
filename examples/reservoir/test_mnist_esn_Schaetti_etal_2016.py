"""
Testing features and method for
Echo State Network - Reservoir for MNIST digit classification

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

epochs = 10
batch_size = 50
num_batches =  int(np.ceil(x_train_num_images / batch_size))
width = 100#1200
image_size = 28
input_size = image_size
memory_size = image_size
input_scaling = 0.6
input_sparsity = 0.9
esn_sparsity = 0.1
bias_scaling = 1.0
output_layer_size = 10
leaky_rate = 0.8 # 0.2 in paper
spectral_radius = 1.3
image_num_pixels = x_train_image_shape[0] * x_train_image_shape[1]

exp = experiment.Experiment(input_start=0,input_delay=0,training_start=image_size,
                            training_delay=image_size,reset_cells_after_train=True,
                            batch_size=batch_size)


input_esn = exp.add_input(tf.float64, [input_size], "input_esn")
desired_output = exp.add_desired_output(tf.float64, [output_layer_size], "desired_output")

g_esn = exp.add_group_cells(name="g_esn", amount=width)
g_esn_real = g_esn.add_real_state(state_name='g_esn_real')

# exp.add_connection("input_conn", connection.IndexConnection(input_esn,g_esn_real,
#                                                             np.arange(width)))

g_input_real_conn = conn_random.create_gaussian_connection('g_input_real_conn',
                                                          input_size, width,
                                                          scale=input_scaling,
                                                          sparsity=input_sparsity,
                                                          is_sparse=False)

# exp.add_connection("input_conn",
#                    connection.WeightedConnection(input_esn,
#                                                  g_esn_real,act.leaky_sigmoid,
#                                                  g_input_real_conn,
#                                                  fargs_list=[(leaky_rate,)]))

exp.add_connection("input_conn",
                   connection.WeightedConnection(input_esn,
                                                 g_esn_real,None,
                                                 g_input_real_conn))


# indices = [[i,i] for i in range(width)]
# values = [1]*width
# dense_shape = [width, width]

# g_esn_real_conn = conn_custom.create_custom_sparse_matrix('g_esn_real_conn',
#                                                           indices,
#                                                           values,
#                                                           dense_shape)

g_esn_real_conn = conn_random.create_gaussian_matrix('g_esn_real_conn',
                                                      width,
                                                      spectral_radius=spectral_radius,
                                                      sparsity=esn_sparsity,
                                                      is_sparse=False)

g_esn_real_bias = conn_random.create_gaussian_connection('g_esn_real_bias',
                                                         1, width,
                                                         scale=bias_scaling,
                                                         is_sparse=False)

exp.add_connection("g_esn_conn",
                    connection.BiasWeightedConnection(g_esn_real,
                                                  g_esn_real,act.leaky_tanh,
                                                  g_esn_real_conn,
                                                  g_esn_real_bias,
                                                  fargs_list=[(leaky_rate,)]))

# exp.add_connection("g_esn_conn",
#                    connection.BiasWeightedConnection(g_esn_real,
#                                                  g_esn_real,act.sigmoid,
#                                                  g_esn_real_conn,
#                                                  g_esn_real_bias))

output_layer =  exp.add_group_cells(name="output_layer", amount=output_layer_size)
output_layer_real_state = output_layer.add_real_state(state_name='output_layer_real_state')


#esn_output_conn = conn_random.create_xavier_connection("esn_output_conn", width, output_layer_size)
g_esn_memory = exp.add_state_memory(g_esn_real,memory_size)
esn_output_conn = conn_random.create_xavier_connection("esn_output_conn", width*memory_size, output_layer_size)

esn_output_bias = conn_random.create_xavier_connection("esn_output_bias", 1, output_layer_size)
# exp.add_trainable_connection("output_conn",
#                              connection.WeightedConnection(g_esn_real,
#                                                            output_layer_real_state,
#                                                            act.sigmoid,
#                                                            esn_output_conn))
exp.add_trainable_connection("output_conn",
                             connection.BiasWeightedConnection(g_esn_memory,
                                                               output_layer_real_state,
                                                               act.sigmoid,
                                                               esn_output_conn,
                                                               esn_output_bias))

c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=exp.trainable_connections["output_conn"].output,
    labels=desired_output,
    axis=0))

exp.set_training(c_loss,0.03)

# Monitors are needed because "reset_cells_after_train=True"
exp.add_monitor("output_layer", "output_layer_real_state", timesteps=1)
exp.add_monitor("g_esn", "g_esn_real", timesteps=1)

exp.initialize_cells()

for epoch in range(epochs):
  print("Epoch:", epoch)
  shuffled_indices = np.random.permutation(x_train_num_images)
  batch_indices = np.split(shuffled_indices,\
                           np.arange(batch_size,x_train_num_images,batch_size))
  for step, batch_idx in enumerate(batch_indices):
    for i in range(image_size):
      input_esn_batch = x_train[i*input_size:(i+1)*input_size,batch_idx]
      desired_output_batch = y_train[:,batch_idx]
      feed_dict = {input_esn: input_esn_batch, desired_output: desired_output_batch}
      exp.run_step(feed_dict=feed_dict)

      prediction_batch = exp.get_monitor("output_layer", "output_layer_real_state")[0,:,:]
      accuracy_batch = np.sum(np.argmax(prediction_batch, axis=0) == np.argmax(desired_output_batch, axis=0)) / batch_size

    utils.progressbar_loss_accu(step+1, num_batches, exp.training_loss, accuracy_batch)
