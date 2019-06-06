""" Cellular automata 1D - Self-organized criticality"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import numpy as np
import time

width = 1000
timesteps = 10000

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx,
                                           is_wrapped_ca=True)

g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
                             activation_func=act.rule_binary_ca_1d_width3_func,\
                             fargs=(110,))

#g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
#                             activation_func=act.rule_binary_ca_1d_width3_func,\
#                             fargs=(10,))

exp.add_monitor("g_ca", "g_ca_bin", timesteps)

exp.initialize_cells()

start = time.time()

exp.run(timesteps=timesteps)
ca_result = exp.get_monitor("g_ca", "g_ca_bin")

print("Execution time:", time.time()-start)

exp.close()

def getdict_cluster_size(arr1d):
  cluster_dict = {}
  current_number = None
  for a in arr1d:
    if current_number == a:
      cluster_dict[a][-1] = cluster_dict[a][-1]+1
    else:
      current_number = a
      if a in cluster_dict:
        cluster_dict[a].append(1)
      else:
        cluster_dict[a] = [1]
  return cluster_dict


def getlist_avalanche_size(x, value):
  x0size, x1size = x.shape
  list_avalance_size = []
  for i in range(x0size):
    list_avalance_size.extend(getdict_cluster_size(x[i,:])[value])
  return list_avalance_size

def getlist_avalanche_count(x, value):
  x0size, x1size = x.shape
  list_avalance_size = []
  for i in range(x0size):
    list_avalance_size.append(list(x[i,:]).count(value))
  return list_avalance_size

def getlist_avalanche_duration(x, value):
  x0size, x1size = x.shape
  list_avalance_duration = []
  for i in range(x1size):
    list_avalance_duration.extend(getdict_cluster_size(x[i,:])[value])
  return list_avalance_duration

def plot_distribution(distribution, title=""):
  import matplotlib.pyplot as plt
  plt.figure()
  x = np.linspace(1,len(distribution),len(distribution))
  plt.loglog(x, distribution, "o-", label="pdf")
  
  plt.loglog(x, [distribution[0]*xx**(-5.) for xx in x], label="slope=-5")
  plt.legend(loc='upper right')
  plt.title(title)
  plt.show()

avalanche_s_0 = getlist_avalanche_size(ca_result, 0)
avalanche_d_0 = getlist_avalanche_duration(ca_result, 0)
#avalanche_c_0 = getlist_avalanche_count(ca_result, 0)
avalanche_s_0_bc = np.bincount(avalanche_s_0)[1:]
avalanche_d_0_bc = np.bincount(avalanche_d_0)[1:]
#avalanche_c_0_bc = np.bincount(avalanche_c_0)[1:]

avalanche_s_1 = getlist_avalanche_size(ca_result, 1)
avalanche_d_1 = getlist_avalanche_duration(ca_result, 1)
#avalanche_c_1 = getlist_avalanche_count(ca_result, 1)
avalanche_s_1_bc = np.bincount(avalanche_s_1)[1:]
avalanche_d_1_bc = np.bincount(avalanche_d_1)[1:]
#avalanche_c_1_bc = np.bincount(avalanche_c_1)[1:]

plot_distribution(avalanche_s_0_bc, "Avalanche size | Elementary CA rule 110 | v=0 | N=10^3 | t=10^4")
plot_distribution(avalanche_d_0_bc, "Avalanche duration | Elementary CA rule 110 | v=0 | N=10^3 | t=10^4")
#plot_distribution(avalanche_c_0_bc, "Avalanche count | Elementary CA rule 30 | v=0 | N=10^3 | t=10^4")
plot_distribution(avalanche_s_1_bc, "Avalanche size | Elementary CA rule 110 | v=1 | N=10^3 | t=10^4")
plot_distribution(avalanche_d_1_bc, "Avalanche duration | Elementary CA rule 110 | v=1 | N=10^3 | t=10^4")
#plot_distribution(avalanche_c_1_bc, "Avalanche count | Elementary CA rule 30 | v=1 | N=10^3 | t=10^4")

np.savez("dict_avalanche_bincount.npz", avalanche_s_0_bc, avalanche_d_0_bc, avalanche_s_1_bc, avalanche_d_1_bc)
