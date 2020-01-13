""" Game of life """

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
#import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os
plt.rcParams.update({'font.size': 18})

width = 5
height = 5
timesteps = 25

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width*height)
neighbors, center_idx = ca.create_count_neighbors_ca2d(3,3)

initial_state = np.array([5*[0],
                          [0,1,0,0,0],
                          [0,0,1,1,0],
                          [0,1,1,0,0],
                          5*[0]])

print(initial_state.shape)

g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init=initial_state)



g_ca_bin_conn = ca.create_conn_matrix_ca2d('g_ca_bin_conn',width,height,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
                             activation_func=act.game_of_life_func)

exp.add_monitor("g_ca", "g_ca_bin")

exp.initialize_cells()

exp.run(timesteps=timesteps)


ca_result = exp.get_monitor("g_ca", "g_ca_bin")

pca = PCA(n_components=2)

principal_comp = pca.fit_transform(ca_result)
min_x_pca = np.min(principal_comp[:,0])
min_y_pca = np.min(principal_comp[:,1])
max_x_pca = np.max(principal_comp[:,0])
max_y_pca = np.max(principal_comp[:,1])


output_folder_path = os.path.join("results", "game_of_life_pca_v2")
if not os.path.exists(output_folder_path):
  os.makedirs(output_folder_path)


for i in range(timesteps):
  fig, axs = plt.subplots(1,2)
  #fig.patch.set_facecolor('xkcd:mint green')
  #fig.subplots_adjust(top = 0.5)
  axs[0].set_xticks([])
  axs[0].set_yticks([])
  axs[1].set_xlim(min_x_pca-.2, max_x_pca+.2)
  axs[1].set_ylim(min_y_pca-.2, max_y_pca+.2)
  axs[1].set_xticks([-1,0,1])
  axs[1].set_yticks([-1,0,1])
  axs[1].set_aspect("equal")

  #axs[0].imshow(1-ca_result[i].reshape((width,height)), cmap="gray")
  #axs[0].grid(color="k", linewidth=1)
  axs[0].pcolormesh(1-np.flipud(ca_result[i].reshape((width,height))), cmap="gray", edgecolors="lightgray", linewidths=1)
  axs[0].set_aspect("equal")
  axs[1].plot(principal_comp[:i+1,0], principal_comp[:i+1,1], c="darkgray", zorder=0)
  axs[1].scatter(principal_comp[:i,0], principal_comp[:i,1], c="darkgray", marker="o", zorder=5)
  axs[1].scatter(principal_comp[i,0], principal_comp[i,1], s=150, c="k", marker="*", zorder=10)

  plt.savefig(os.path.join(output_folder_path, "img_"+str(i+1)+".png"), bbox_inches = 'tight', pad_inches=0.1)
  plt.savefig(os.path.join(output_folder_path, "s"+str("%02d" % (i+1))+".eps"), bbox_inches = 'tight', pad_inches=0.1)

  plt.close('all')

