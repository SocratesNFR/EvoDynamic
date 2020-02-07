""" Cellular automata 1D - animation """

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np
import time

width = 100
height_fig = 200
timesteps = 400

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

#fargs_list = [(a,) for a in [246, 131]]

#fargs_list = [(a,) for a in [54, 52, 144, 127, 30, 17, 159, 81, 75, 226, 222,\
#              16, 191, 206, 14, 133, 237, 191, 204, 92, 98, 8, 55, 202, 169,\
#              243, 221, 80, 102, 154, 186, 125, 2, 172, 237, 242, 184, 140, 208,\
#              34, 248]]

#[235, 36]
#fargs_list = [(a,) for a in [235, 36]]
fargs_list = [(a,) for a in [110]]


exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))

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

#plt.show()

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save('rule110_'+time.strftime("%Y%m%d-%H%M%S")+'.mp4', writer=writer)

plt.connect('close_event', exp.close())