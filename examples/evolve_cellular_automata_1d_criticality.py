""" Evolving Cellular automata 1D - Self-organized criticality"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np
from sklearn.linear_model import LinearRegression
import time

width = 100
timesteps = 1000

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

def evaluate_result(ca_result):
  avalanche_s_0 = getlist_avalanche_size(ca_result, 0)
  avalanche_d_0 = getlist_avalanche_duration(ca_result, 0)
  avalanche_s_0_bc = np.bincount(avalanche_s_0)[1:]
  avalanche_d_0_bc = np.bincount(avalanche_d_0)[1:]
  
  avalanche_s_1 = getlist_avalanche_size(ca_result, 1)
  avalanche_d_1 = getlist_avalanche_duration(ca_result, 1)
  avalanche_s_1_bc = np.bincount(avalanche_s_1)[1:]
  avalanche_d_1_bc = np.bincount(avalanche_d_1)[1:]

  log_avalanche_s_0_bc = np.log10(avalanche_s_0_bc)
  log_avalanche_d_0_bc = np.log10(avalanche_d_0_bc)
  log_avalanche_s_1_bc = np.log10(avalanche_s_1_bc)
  log_avalanche_d_1_bc = np.log10(avalanche_d_1_bc)
  
  log_avalanche_s_0_bc = np.where(np.isfinite(log_avalanche_s_0_bc), log_avalanche_s_0_bc, 0)
  log_avalanche_d_0_bc = np.where(np.isfinite(log_avalanche_d_0_bc), log_avalanche_d_0_bc, 0)
  log_avalanche_s_1_bc = np.where(np.isfinite(log_avalanche_s_1_bc), log_avalanche_s_1_bc, 0)
  log_avalanche_d_1_bc = np.where(np.isfinite(log_avalanche_d_1_bc), log_avalanche_d_1_bc, 0)
  
  fit_avalanche_s_0_bc = LinearRegression().fit(np.arange(1,len(avalanche_s_0_bc)+1).reshape(-1,1), log_avalanche_s_0_bc)
  fit_avalanche_d_0_bc = LinearRegression().fit(np.arange(1,len(avalanche_d_0_bc)+1).reshape(-1,1), log_avalanche_d_0_bc)
  fit_avalanche_s_1_bc = LinearRegression().fit(np.arange(1,len(avalanche_s_1_bc)+1).reshape(-1,1), log_avalanche_s_1_bc)
  fit_avalanche_d_1_bc = LinearRegression().fit(np.arange(1,len(avalanche_d_1_bc)+1).reshape(-1,1), log_avalanche_d_1_bc)
  
  lin_err = []
  lin_err.append(fit_avalanche_s_0_bc.score(np.arange(1,len(avalanche_s_0_bc)+1).reshape(-1,1), log_avalanche_s_0_bc))
  lin_err.append(fit_avalanche_d_0_bc.score(np.arange(1,len(avalanche_d_0_bc)+1).reshape(-1,1), log_avalanche_d_0_bc))
  lin_err.append(fit_avalanche_s_1_bc.score(np.arange(1,len(avalanche_s_1_bc)+1).reshape(-1,1), log_avalanche_s_1_bc))
  lin_err.append(fit_avalanche_d_1_bc.score(np.arange(1,len(avalanche_d_1_bc)+1).reshape(-1,1), log_avalanche_d_1_bc))
  print(lin_err)

  coef = []
  coef.append(fit_avalanche_s_0_bc.coef_[0])
  coef.append(fit_avalanche_d_0_bc.coef_[0])
  coef.append(fit_avalanche_s_1_bc.coef_[0])
  coef.append(fit_avalanche_d_1_bc.coef_[0])
  print(coef)

def evolve_experiment(gen=10, pop=10):
  rule_list = [10, 30, 110, 90]
  ca_result = []
  for rule in rule_list:
  #for g in range(gen):
    exp = experiment.Experiment()
    g_ca = exp.add_group_cells(name="g_ca", amount=width)
    neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
    g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
    g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                               neighbors=neighbors,\
                                               center_idx=center_idx,
                                               is_wrapped_ca=True)


    exp.add_connection("g_ca_conn", connection.WeightedConnection(g_ca_bin,g_ca_bin,act.rule_binary_ca_1d_width3_func,g_ca_bin_conn, fargs_list=[(rule,)]))

    exp.add_monitor("g_ca", "g_ca_bin", timesteps)

    exp.initialize_cells()

    start = time.time()

    exp.run(timesteps=timesteps)
    ca_result.append(exp.get_monitor("g_ca", "g_ca_bin"))

    print("Execution time:", time.time()-start)

    exp.close()
  return np.array(ca_result)

start_total = time.time()

ca_res = evolve_experiment()

print("TOTAL Execution time:", time.time()-start_total)

evaluate_result(ca_res[0])
evaluate_result(ca_res[1])
evaluate_result(ca_res[2])
evaluate_result(ca_res[3])