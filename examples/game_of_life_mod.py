""" Game of life """

import tensorflow as tf
import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
#import time

width = 100
height = 75

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width*height)
neighbors, center_idx = ca.create_count_neighbors_ca2d(3,3)

neighbors_mod = neighbors.astype(np.float64)

neighbors_mod[0,1] += 0.1
neighbors_mod[1,0] += 0.1
neighbors_mod[1,2] += 0.1
neighbors_mod[2,1] += 0.1

print(neighbors_mod)

g_ca.add_binary_state(state_name='g_ca_bin')

g_ca_bin_conn = ca.create_conn_matrix_ca2d('g_ca_bin_conn',width,height,\
                                           neighbors=neighbors_mod,\
                                           center_idx=center_idx)

# Modification of function in evodynamic.activation
def game_of_life_mod_func(count_neighbors, previous_state):
  count_neighbors_round = tf.round(count_neighbors)

  born_cells_op = tf.logical_or(tf.equal(count_neighbors_round, 3),
                                tf.logical_and(tf.equal(count_neighbors_round, 4),
                                               tf.greater(count_neighbors, 4.35)))


  born_cells_op = tf.logical_or(tf.logical_or(tf.equal(count_neighbors_round, 3),
                              tf.logical_and(tf.equal(count_neighbors_round, 4),
                                             tf.greater(count_neighbors, 4.35))),
                                tf.logical_and(tf.equal(count_neighbors_round, 4),
                                             tf.less(count_neighbors, 4.05)))

  kill_cells_sub_op = tf.less(count_neighbors_round, 2)

  kill_cells_over_op = tf.logical_and(tf.greater(count_neighbors_round, 3),
                                      tf.logical_not(tf.logical_and(
                                          tf.equal(count_neighbors_round, 4),
                                          tf.logical_or(
                                              tf.greater(count_neighbors, 4.35),
                                              tf.less(count_neighbors, 4.05)))))

  kill_cells_op = tf.logical_or(kill_cells_sub_op, kill_cells_over_op)

  update_kill_op = tf.where(kill_cells_op, tf.zeros(tf.shape(previous_state), dtype=tf.float64), previous_state)

  # Return update_born_op
  return tf.where(born_cells_op, tf.ones(tf.shape(previous_state), dtype=tf.float64), update_kill_op)

g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
                             activation_func=game_of_life_mod_func)



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
    exp.run_step()
    im.set_array(exp.get_group_cells_state("g_ca", "g_ca_bin").reshape((height,width)))

    plt.title('Step: '+str(idx_anim))
    idx_anim += 1
    return im,# ttl

ani = animation.FuncAnimation(fig, updatefig, frames=300, interval=200, blit=False)

plt.show()

# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#ani.save('results/gameoflifemod_'+time.strftime("%Y%m%d-%H%M%S")+'.mp4', writer=writer)

plt.connect('close_event', exp.close())