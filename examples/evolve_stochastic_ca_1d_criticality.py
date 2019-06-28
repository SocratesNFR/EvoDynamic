""" Evolving Stochastic Cellular automata 1D - Self-organized criticality"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
from evodynamic.evolution import ga
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
  list_avalance_size = []
  if value in x:
    x0size, x1size = x.shape
    for i in range(x0size):
      if value in x[i,:]:
        list_avalance_size.extend(getdict_cluster_size(x[i,:])[value])
  return list_avalance_size

def getlist_avalanche_count(x, value):
  list_avalance_count = []
  if value in x:
    x0size, x1size = x.shape  
    for i in range(x0size):
      if value in x[i,:]:
        list_avalance_count.append(list(x[i,:]).count(value))
  return list_avalance_count

def getlist_avalanche_duration(x, value):
  list_avalance_duration = []
  if value in x:
    x0size, x1size = x.shape
    for i in range(x1size):
      if value in x[:,i]:
        list_avalance_duration.extend(getdict_cluster_size(x[:,i])[value])
  return list_avalance_duration

def norm_coef(coef):
  return -coef * 0.1

def norm_diff(arr):
  adiff = np.diff(np.diff(arr))
  return 1+np.mean(adiff)+np.median(adiff)

def evaluate_result(ca_result):
  avalanche_s_0 = getlist_avalanche_size(ca_result, 0)
  avalanche_d_0 = getlist_avalanche_duration(ca_result, 0)
  avalanche_s_0_bc = np.bincount(avalanche_s_0)[1:] if len(avalanche_s_0) > 5 else []
  avalanche_d_0_bc = np.bincount(avalanche_d_0)[1:] if len(avalanche_d_0) > 5 else []
  
  avalanche_s_1 = getlist_avalanche_size(ca_result, 1)
  avalanche_d_1 = getlist_avalanche_duration(ca_result, 1)
  avalanche_s_1_bc = np.bincount(avalanche_s_1)[1:] if len(avalanche_s_1) > 5 else []
  avalanche_d_1_bc = np.bincount(avalanche_d_1)[1:] if len(avalanche_d_1) > 5 else []

  log_avalanche_s_0_bc = np.log10(avalanche_s_0_bc)
  log_avalanche_d_0_bc = np.log10(avalanche_d_0_bc)
  log_avalanche_s_1_bc = np.log10(avalanche_s_1_bc)
  log_avalanche_d_1_bc = np.log10(avalanche_d_1_bc)
  
  log_avalanche_s_0_bc = np.where(np.isfinite(log_avalanche_s_0_bc), log_avalanche_s_0_bc, 0)
  log_avalanche_d_0_bc = np.where(np.isfinite(log_avalanche_d_0_bc), log_avalanche_d_0_bc, 0)
  log_avalanche_s_1_bc = np.where(np.isfinite(log_avalanche_s_1_bc), log_avalanche_s_1_bc, 0)
  log_avalanche_d_1_bc = np.where(np.isfinite(log_avalanche_d_1_bc), log_avalanche_d_1_bc, 0)

  fitness = 0

  if len(avalanche_s_0_bc) > 5 and len(avalanche_d_0_bc) > 5 and\
    len(avalanche_s_1_bc) > 5 and len(avalanche_d_1_bc) > 5:
    fit_avalanche_s_0_bc = LinearRegression().fit(np.arange(1,len(avalanche_s_0_bc)+1).reshape(-1,1), log_avalanche_s_0_bc)
    fit_avalanche_d_0_bc = LinearRegression().fit(np.arange(1,len(avalanche_d_0_bc)+1).reshape(-1,1), log_avalanche_d_0_bc)
    fit_avalanche_s_1_bc = LinearRegression().fit(np.arange(1,len(avalanche_s_1_bc)+1).reshape(-1,1), log_avalanche_s_1_bc)
    fit_avalanche_d_1_bc = LinearRegression().fit(np.arange(1,len(avalanche_d_1_bc)+1).reshape(-1,1), log_avalanche_d_1_bc)
    
    lin_err = []
    lin_err.append(fit_avalanche_s_0_bc.score(np.arange(1,len(avalanche_s_0_bc)+1).reshape(-1,1), log_avalanche_s_0_bc))
    lin_err.append(fit_avalanche_d_0_bc.score(np.arange(1,len(avalanche_d_0_bc)+1).reshape(-1,1), log_avalanche_d_0_bc))
    lin_err.append(fit_avalanche_s_1_bc.score(np.arange(1,len(avalanche_s_1_bc)+1).reshape(-1,1), log_avalanche_s_1_bc))
    lin_err.append(fit_avalanche_d_1_bc.score(np.arange(1,len(avalanche_d_1_bc)+1).reshape(-1,1), log_avalanche_d_1_bc))
    #print(lin_err)
  
    coef = []
    coef.append(fit_avalanche_s_0_bc.coef_[0])
    coef.append(fit_avalanche_d_0_bc.coef_[0])
    coef.append(fit_avalanche_s_1_bc.coef_[0])
    coef.append(fit_avalanche_d_1_bc.coef_[0])
    #print(coef)
    
    norm_max_avalanche = np.log10(len(avalanche_s_0_bc)+len(avalanche_d_0_bc)+\
      len(avalanche_s_1_bc)+len(avalanche_d_1_bc))

    norm_unique_states = ((np.unique(ca_result, axis=0).shape[0]) / ca_result.shape[1])
    print("norm_max_avalanche", norm_unique_states)
    print("norm_unique_states", norm_unique_states)
    fitness = np.mean(lin_err) + norm_coef(np.mean(coef))+norm_max_avalanche+norm_unique_states#+norm_diff(log_avalanche_s_0_bc)\
      #+norm_diff(log_avalanche_d_0_bc)+norm_diff(log_avalanche_s_1_bc)+norm_diff(log_avalanche_d_1_bc)
  print("Fitness", fitness)
  return fitness

# genome is a list of integers between 0 and 255
def evaluate_genome(genome=8*[0.5]):
  print(genome)
  gen_rule = [(genome,)]
  
  exp = experiment.Experiment()
  g_ca = exp.add_group_cells(name="g_ca", amount=width)
  neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
  g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
  g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                             neighbors=neighbors,\
                                             center_idx=center_idx,
                                             is_wrapped_ca=True)


  exp.add_connection("g_ca_conn",
                     connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                   act.rule_binary_sca_1d_width3_func,
                                                   g_ca_bin_conn, fargs_list=gen_rule))

  exp.add_monitor("g_ca", "g_ca_bin", timesteps)

  exp.initialize_cells()

  start = time.time()

  exp.run(timesteps=timesteps)
  #ca_result .append()

  print("Execution time:", time.time()-start)

  exp.close()
  return evaluate_result(exp.get_monitor("g_ca", "g_ca_bin"))

start_total = time.time()

best_genome = ga.evolve_probability(evaluate_genome, pop_size=20, generation=20)

print("TOTAL Execution time:", time.time()-start_total)

print(best_genome)
print("Final fitness", evaluate_genome(best_genome))
print("Final fitness", evaluate_genome(best_genome))
print("Final fitness", evaluate_genome(best_genome))
print("Final fitness", evaluate_genome(best_genome))

