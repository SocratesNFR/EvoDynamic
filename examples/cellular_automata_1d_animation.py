""" Cellular automata 1D - animation """

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import numpy as np

width = 100
height_fig = 200
timesteps = 400

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
                             activation_func=act.rule_binary_ca_1d_width3_func,\
                             fargs=(30,))

g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
                             activation_func=act.rule_binary_ca_1d_width3_func,\
                             fargs=(10,))

exp.add_monitor("g_ca", "g_ca_bin")

exp.initialize_cells()

# Animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

idx_anim = 0
im_ca = np.zeros((height_fig,width))
im_ca[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")

im = plt.imshow(im_ca, animated=True)

plt.title('Step: 0')

def updatefig(*args):
    global idx_anim, im_ca
    idx_anim += 1
    exp.run_step()
    if idx_anim < height_fig:
      im_ca[idx_anim] = exp.get_group_cells_state("g_ca", "g_ca_bin")
      im.set_array(im_ca)
    else:
      im_ca = np.vstack((im_ca[1:], exp.get_group_cells_state("g_ca", "g_ca_bin")))
      im.set_array(im_ca)

    plt.title('Step: '+str(idx_anim+1))

    return im,# ttl

ani = animation.FuncAnimation(fig, updatefig, frames=timesteps-2,\
                              interval=100, blit=False, repeat=False)

plt.show()
plt.connect('close_event', exp.close())