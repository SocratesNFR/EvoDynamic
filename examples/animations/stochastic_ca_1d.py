""" Cellular automata 1D - animation """

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np

width = 100
height_fig = 200
timesteps = 400

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init="central")
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

#fargs_list = [([0.8653582694285038, 0.07574324505979713, 0.0,\
#                0.6701419392338681, 0.02243120000370638, 0.46365825379340864,\
#                0.7473194740998363, 1.0],)]


#fargs_list = [([0.10300948029035227, 0.5367869451270947, 0.2167946269388179,\
#                0.3934686667827154, 0.6798368229670028, 0.17545801387723042,\
#                0.724778917324477, 1.0],)]

"""
Best 2
80;4.375510397876782;{'norm_ksdist_res': 0.9613162536023352,
'norm_coef_res': 1.6377095314393233, 'norm_unique_states': 1.0,
'norm_avalanche_pdf_size': 0.9659114597767141, 'norm_linscore_res': 0.8704469695084798,
'norm_R_res': 0.7277920719335422, 'fitness': 4.375510397876782};
[0.39422172047670734, 0.09472197628905793, 0.2394927250526252, 0.4084554505943707, 0.0, 0.7302038703441202, 0.9150343715586952, 1.0]
"""
fargs_list = [( [0.39422172047670734, 0.09472197628905793,\
                 0.2394927250526252, 0.4084554505943707, 0.0,\
                 0.7302038703441202, 0.9150343715586952, 1.0],)]


exp.add_connection("g_ca_conn",
                     connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                   act.rule_binary_sca_1d_width3_func,
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

plt.show()
plt.connect('close_event', exp.close())