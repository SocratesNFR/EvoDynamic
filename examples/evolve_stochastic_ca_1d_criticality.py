""" Evolving Stochastic Cellular automata 1D - Self-organized criticality"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
from evodynamic.evolution import ga
import numpy as np
#from sklearn.linear_model import LinearRegression
import time
import powerlaw

width = 1000
timesteps = 1000

#def KSdist(theoretical_pdf, empirical_pdf):
#  return np.max(np.abs(np.cumsum(theoretical_pdf) - np.cumsum(empirical_pdf)))

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

def getarray_avalanche_size(x, value):
  list_avalance_size = []
  if value in x:
    x0size, x1size = x.shape
    for i in range(x0size):
      if value in x[i,:]:
        list_avalance_size.extend(getdict_cluster_size(x[i,:])[value])
  return np.array(list_avalance_size)

def getarray_avalanche_duration(x, value):
  list_avalance_duration = []
  if value in x:
    x0size, x1size = x.shape
    for i in range(x1size):
      if value in x[:,i]:
        list_avalance_duration.extend(getdict_cluster_size(x[:,i])[value])
  return np.array(list_avalance_duration)

def norm_alpha(alpha):
  return 0.1*np.mean(alpha)

# Normalize values from 0 to inf to be from 10 to 0
def norm_ksdist(ksdist, smooth=0.1):
  return np.exp(-smooth * (np.min(ksdist)+0.1*np.mean(ksdist)))

# Normalize values from -inf to inf to be from 0 to 1
def norm_R(R, smooth=0.01):
  return 10 / (1+np.exp(-smooth * (np.max(R)+0.1*np.mean(R))))

def calculate_data_score(data):
  fit = powerlaw.Fit(data, xmin =1, discrete= True)
  alpha = fit.power_law.alpha
  ksdist = fit.power_law.D
  R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
  R_exp = R_exp if p_exp < 0.1 else 0
  R_log, p_log = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
  R_log = R_log if p_log < 0.1 else 0
  R = R_exp+R_log

  return alpha, ksdist, R

def evaluate_result(ca_result):
  avalanche_s_0 = getarray_avalanche_size(ca_result, 0)
  avalanche_d_0 = getarray_avalanche_duration(ca_result, 0)
  avalanche_s_1 = getarray_avalanche_size(ca_result, 1)
  avalanche_d_1 = getarray_avalanche_duration(ca_result, 1)

  fit_avalanche_s_0 = calculate_data_score(avalanche_s_0)
  fit_avalanche_d_0 = calculate_data_score(avalanche_d_0)
  fit_avalanche_s_1 = calculate_data_score(avalanche_s_1)
  fit_avalanche_d_1 = calculate_data_score(avalanche_d_1)

  alpha_list = [fit_avalanche_s_0[0], fit_avalanche_d_0[0], fit_avalanche_s_1[0],\
                fit_avalanche_d_1[0]]
  ksdist_list = [fit_avalanche_s_0[1], fit_avalanche_d_0[1], fit_avalanche_s_1[1],\
                fit_avalanche_d_1[1]]
  R_list = [fit_avalanche_s_0[2], fit_avalanche_d_0[2], fit_avalanche_s_1[2],\
            fit_avalanche_d_1[2]]
  
  print("alpha_list", alpha_list)
  print("ksdist_list", ksdist_list)
  print("R_list", R_list)
  
  norm_ksdist_res = norm_ksdist(ksdist_list)
  norm_alpha_res = norm_alpha(alpha_list)
  norm_R_res = norm_R(R_list)
  norm_unique_states = ((np.unique(ca_result, axis=0).shape[0]) / ca_result.shape[1])

  print("norm_ksdist_res", norm_ksdist_res)
  print("norm_alpha_res", norm_alpha_res)
  print("norm_R_res", norm_R_res)
  print("norm_unique_states", norm_unique_states)

  fitness = norm_ksdist_res + norm_alpha_res + norm_R_res + norm_unique_states
  print("Fitness", fitness)
  return fitness

# genome is a list of float numbers between 0 and 1
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

best_genome = ga.evolve_probability(evaluate_genome, pop_size=10, generation=10)

print("TOTAL Execution time:", time.time()-start_total)

print(best_genome)
print("Final fitness", evaluate_genome(best_genome))
print("Final fitness", evaluate_genome(best_genome))
print("Final fitness", evaluate_genome(best_genome))
print("Final fitness", evaluate_genome(best_genome))

