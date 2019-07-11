""" Evolving Cellular automata 1D - Self-organized criticality"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
from evodynamic.evolution import ga
from sklearn.linear_model import LinearRegression
import numpy as np
import time
import powerlaw
import csv
import os
import sys

width = 1000
timesteps = 1000

def KSdist(theoretical_pdf, empirical_pdf):
  return np.max(np.abs(np.cumsum(theoretical_pdf) - np.cumsum(empirical_pdf)))

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

def norm_coef(coef):
  return -0.1*np.mean(coef)

def norm_alpha(alpha):
  return 0.1*np.mean(alpha)

def norm_linscore(linscore):
  return 10*np.mean(linscore)#5*np.max(linscore)+5*np.mean(linscore)

# Normalize values from 0 to inf to be from 10 to 0
def norm_ksdist(ksdist, smooth=1):
  return 10*np.exp(-smooth * (np.min(ksdist)+0.1*np.mean(ksdist)))

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

def evaluate_result(ca_result, filename=None):
  avalanche_s_0 = getarray_avalanche_size(ca_result, 0)
  avalanche_d_0 = getarray_avalanche_duration(ca_result, 0)
  avalanche_s_0_bc = np.bincount(avalanche_s_0)[1:] if len(avalanche_s_0) > 5 else []
  avalanche_d_0_bc = np.bincount(avalanche_d_0)[1:] if len(avalanche_d_0) > 5 else []
  
  avalanche_s_1 = getarray_avalanche_size(ca_result, 1)
  avalanche_d_1 = getarray_avalanche_duration(ca_result, 1)
  avalanche_s_1_bc = np.bincount(avalanche_s_1)[1:] if len(avalanche_s_1) > 5 else []
  avalanche_d_1_bc = np.bincount(avalanche_d_1)[1:] if len(avalanche_d_1) > 5 else []

  avalanche_s_0_bc = avalanche_s_0_bc/sum(avalanche_s_0_bc)
  avalanche_d_0_bc = avalanche_d_0_bc/sum(avalanche_d_0_bc)
  avalanche_s_1_bc = avalanche_s_1_bc/sum(avalanche_s_1_bc)
  avalanche_d_1_bc = avalanche_d_1_bc/sum(avalanche_d_1_bc)

  mask_avalanche_s_0_bc = avalanche_s_0_bc > 0
  mask_avalanche_d_0_bc = avalanche_d_0_bc > 0
  mask_avalanche_s_1_bc = avalanche_s_1_bc > 0
  mask_avalanche_d_1_bc = avalanche_d_1_bc > 0

  log_avalanche_s_0_bc = np.log10(avalanche_s_0_bc)
  log_avalanche_d_0_bc = np.log10(avalanche_d_0_bc)
  log_avalanche_s_1_bc = np.log10(avalanche_s_1_bc)
  log_avalanche_d_1_bc = np.log10(avalanche_d_1_bc)
  
  log_avalanche_s_0_bc = np.where(mask_avalanche_s_0_bc, log_avalanche_s_0_bc, 0)
  log_avalanche_d_0_bc = np.where(mask_avalanche_d_0_bc, log_avalanche_d_0_bc, 0)
  log_avalanche_s_1_bc = np.where(mask_avalanche_s_1_bc, log_avalanche_s_1_bc, 0)
  log_avalanche_d_1_bc = np.where(mask_avalanche_d_1_bc, log_avalanche_d_1_bc, 0)

  fitness = 0
  norm_max_avalanche = 0
  norm_linscore_res = 0
  norm_ksdist_res = 0
  norm_coef_res = 0
  norm_unique_states = 0

  if sum(mask_avalanche_s_0_bc[:10]) > 5 and sum(mask_avalanche_d_0_bc[:10]) > 5 and\
    sum(mask_avalanche_s_1_bc[:10]) > 5 and sum(mask_avalanche_d_1_bc[:10]) > 5:

    # Fit PDF using least square error
    fit_avalanche_s_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_0_bc)+1)[mask_avalanche_s_0_bc]).reshape(-1,1), log_avalanche_s_0_bc[mask_avalanche_s_0_bc])
    fit_avalanche_d_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_0_bc)+1)[mask_avalanche_d_0_bc]).reshape(-1,1), log_avalanche_d_0_bc[mask_avalanche_d_0_bc])
    fit_avalanche_s_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_1_bc)+1)[mask_avalanche_s_1_bc]).reshape(-1,1), log_avalanche_s_1_bc[mask_avalanche_s_1_bc])
    fit_avalanche_d_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_1_bc)+1)[mask_avalanche_d_1_bc]).reshape(-1,1), log_avalanche_d_1_bc[mask_avalanche_d_1_bc])

    linscore_list = []
    linscore_list.append(fit_avalanche_s_0_bc.score(np.log10(np.arange(1,len(avalanche_s_0_bc)+1)[mask_avalanche_s_0_bc]).reshape(-1,1), log_avalanche_s_0_bc[mask_avalanche_s_0_bc]))
    linscore_list.append(fit_avalanche_d_0_bc.score(np.log10(np.arange(1,len(avalanche_d_0_bc)+1)[mask_avalanche_d_0_bc]).reshape(-1,1), log_avalanche_d_0_bc[mask_avalanche_d_0_bc]))
    linscore_list.append(fit_avalanche_s_1_bc.score(np.log10(np.arange(1,len(avalanche_s_1_bc)+1)[mask_avalanche_s_1_bc]).reshape(-1,1), log_avalanche_s_1_bc[mask_avalanche_s_1_bc]))
    linscore_list.append(fit_avalanche_d_1_bc.score(np.log10(np.arange(1,len(avalanche_d_1_bc)+1)[mask_avalanche_d_1_bc]).reshape(-1,1), log_avalanche_d_1_bc[mask_avalanche_d_1_bc]))

    # Fit PDF using least square error
    fit_avalanche_s_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_0_bc)+1)[mask_avalanche_s_0_bc]).reshape(-1,1), log_avalanche_s_0_bc[mask_avalanche_s_0_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_s_0_bc))[mask_avalanche_s_0_bc]])
    fit_avalanche_d_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_0_bc)+1)[mask_avalanche_d_0_bc]).reshape(-1,1), log_avalanche_d_0_bc[mask_avalanche_d_0_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_d_0_bc))[mask_avalanche_d_0_bc]])
    fit_avalanche_s_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_1_bc)+1)[mask_avalanche_s_1_bc]).reshape(-1,1), log_avalanche_s_1_bc[mask_avalanche_s_1_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_s_1_bc))[mask_avalanche_s_1_bc]])
    fit_avalanche_d_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_1_bc)+1)[mask_avalanche_d_1_bc]).reshape(-1,1), log_avalanche_d_1_bc[mask_avalanche_d_1_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_d_1_bc))[mask_avalanche_d_1_bc]])

    theor_avalanche_s_0_bc = np.power(10,fit_avalanche_s_0_bc.predict(np.log10(np.arange(1,len(avalanche_s_0_bc)+1).reshape(-1,1))))
    theor_avalanche_d_0_bc = np.power(10,fit_avalanche_d_0_bc.predict(np.log10(np.arange(1,len(avalanche_d_0_bc)+1).reshape(-1,1))))
    theor_avalanche_s_1_bc = np.power(10,fit_avalanche_s_1_bc.predict(np.log10(np.arange(1,len(avalanche_s_1_bc)+1).reshape(-1,1))))
    theor_avalanche_d_1_bc = np.power(10,fit_avalanche_d_1_bc.predict(np.log10(np.arange(1,len(avalanche_d_1_bc)+1).reshape(-1,1))))

    ksdist_list = []
    ksdist_list.append(KSdist(theor_avalanche_s_0_bc, avalanche_s_0_bc))
    ksdist_list.append(KSdist(theor_avalanche_d_0_bc, avalanche_d_0_bc))
    ksdist_list.append(KSdist(theor_avalanche_s_1_bc, avalanche_s_1_bc))
    ksdist_list.append(KSdist(theor_avalanche_d_1_bc, avalanche_d_1_bc))

    coef_list = []
    coef_list.append(fit_avalanche_s_0_bc.coef_[0])
    coef_list.append(fit_avalanche_d_0_bc.coef_[0])
    coef_list.append(fit_avalanche_s_1_bc.coef_[0])
    coef_list.append(fit_avalanche_d_1_bc.coef_[0])
    #print(coef)

    norm_max_avalanche = 10*np.mean([sum(mask_avalanche_s_0_bc)/width,sum(mask_avalanche_d_0_bc)/timesteps,\
      sum(mask_avalanche_s_1_bc)/width,sum(mask_avalanche_d_1_bc)/timesteps])

    print("linscore_list", linscore_list)
    print("coef_list", coef_list)
    print("ksdist_list", ksdist_list)

    norm_linscore_res = norm_linscore(linscore_list)
    norm_ksdist_res = norm_ksdist(ksdist_list)
    norm_coef_res = norm_coef(coef_list)
    norm_unique_states = 10*((np.unique(ca_result, axis=0).shape[0]) / ca_result.shape[1])


    print("norm_max_avalanche", norm_max_avalanche)
    print("norm_linscore_res", norm_linscore_res)
    print("norm_ksdist_res", norm_ksdist_res)
    print("norm_coef_res", norm_coef_res)
    print("norm_unique_states", norm_unique_states)

    fitness = norm_ksdist_res + norm_coef_res + norm_unique_states + norm_max_avalanche + norm_linscore_res

  val_dict = {}
  val_dict["norm_ksdist_res"] = norm_ksdist_res
  val_dict["norm_coef_res"] = norm_coef_res
  val_dict["norm_unique_states"] = norm_unique_states
  val_dict["norm_max_avalanche"] = norm_max_avalanche
  val_dict["norm_linscore_res"] = norm_linscore_res
  val_dict["fitness"] = fitness

  print("Fitness", fitness)
  return fitness, val_dict

# genome is a list of integers between 0 and 255
def evaluate_genome(genome=[110], filename = None):
  print(genome)
  gen_rule = [(r,) for r in genome]

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
                                                   act.rule_binary_ca_1d_width3_func,
                                                   g_ca_bin_conn, fargs_list=gen_rule))

  exp.add_monitor("g_ca", "g_ca_bin", timesteps)

  exp.initialize_cells()

  start = time.time()

  exp.run(timesteps=timesteps)
  #ca_result .append()

  print("Execution time:", time.time()-start)

  exp.close()
  
  fitness, val_dict = evaluate_result(exp.get_monitor("g_ca", "g_ca_bin"))

  if isinstance(filename, str):
    if ".csv" in filename:
      with open(filename, "a", newline="") as f:
        wr = csv.writer(f, delimiter=";")
        if os.stat(filename).st_size == 0:
          wr.writerow(["genome", "fitness", "norm_ksdist_res", "norm_coef_res", "norm_unique_states",\
                        "norm_max_avalanche", "norm_linscore_res"])

        wr.writerow([str(list(genome)), val_dict["fitness"], val_dict["norm_ksdist_res"],\
                     val_dict["norm_coef_res"], val_dict["norm_unique_states"],\
                     val_dict["norm_max_avalanche"],val_dict["norm_linscore_res"]])
  
  
  return fitness

start_total = time.time()

best_genome = ga.evolve_rules(evaluate_genome, pop_size=2, generation=2)

print("TOTAL Execution time:", time.time()-start_total)

print(best_genome)

print("Final fitness", evaluate_genome(best_genome, sys.argv[1]))

