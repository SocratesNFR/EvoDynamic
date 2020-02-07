""" Cellular automata 1D - Self-organized criticality"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
import powerlaw
from PIL import Image

width = 1000
timesteps = 1000

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init="random")
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx,
                                           is_wrapped_ca=True)

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

exp.add_monitor("g_ca", "g_ca_bin", timesteps)

exp.initialize_cells()

start = time.time()

exp.run(timesteps=timesteps)
ca_result = exp.get_monitor("g_ca", "g_ca_bin")

print("Execution time:", time.time()-start)

exp.close()

def KSdist(theoretical_pdf, empirical_pdf):
  return np.max(np.abs(np.cumsum(theoretical_pdf) - np.cumsum(empirical_pdf)))

def goodness_of_fit(fit, data, gen_data=1000, data_samples_lb=10000):
  theoretical_distribution = powerlaw.Power_Law(xmin=1,\
                                                parameters=[fit.power_law.alpha],\
                                                discrete=True)
  simulated_ksdist_list = []
  data_samples = max(len(data), data_samples_lb)
  print("GoF data_samples", data_samples)
  for _ in range(gen_data):
    simulated_data=theoretical_distribution.generate_random(data_samples)
    simulated_ksdist = powerlaw.power_law_ks_distance(simulated_data,\
                                                      fit.power_law.alpha,\
                                                      xmin=1, xmax=1000, discrete=True)
    simulated_ksdist_list.append(simulated_ksdist)

  return sum(np.array(simulated_ksdist_list) > fit.power_law.D) / gen_data


def powerlaw_stats(data, fname=""):
  fit = powerlaw.Fit(data, xmin =1, discrete= True)
  print()
  print("alpha", fit.power_law.alpha)
  print("xmin", fit.power_law.xmin)
  print("sigma", fit.power_law.sigma)
  print("KSdist", fit.power_law.D)
  print("fit.distribution_compare('power_law', 'exponential')", fit.distribution_compare('power_law', 'exponential', normalized_ratio=True))
  print("fit.distribution_compare('power_law', 'lognormal')", fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True))
  gof = goodness_of_fit(fit, data)
  print("goodness_of_fit(fit, data)", gof)
  print()
  fig, ax = plt.subplots()



  fit.plot_pdf(color = "b", linewidth =2, ax =ax, label="Avalanche (samples=%d)"% len(data))
  fit.power_law.plot_pdf(color = "k", linestyle = "--", ax =ax, label=r"Fit ($\hat{\alpha}$="+"%.1f, $p$-value=%.2f)" % (fit.power_law.alpha, gof))

  ax.legend()
  ax.set_xlabel("$x$")
  ax.set_ylabel("$P(x)$")
  if fname:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig("results/"+fname+timestr+".png", format="png")



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
  return -np.mean(coef)

def norm_linscore(linscore):
  return np.mean(linscore)#5*np.max(linscore)+5*np.mean(linscore)

# Normalize values from 0 to inf to be from 10 to 0
def norm_ksdist(ksdist, smooth=1):
  return np.exp(-smooth * (0.9*np.min(ksdist)+0.1*np.mean(ksdist)))

# Normalize values from -inf to inf to be from 0 to 1
def norm_R(R, smooth=0.01):
  return 1. / (1.+np.exp(-smooth * (0.9*np.max(R)+0.1*np.mean(R))))

def normalize_avalanche_pdf_size(mask_avalanche_s_0_bc, mask_avalanche_d_0_bc,\
                                 mask_avalanche_s_1_bc, mask_avalanche_d_1_bc):
  norm_avalanche_pdf_size_s_0 = sum(mask_avalanche_s_0_bc)/width
  norm_avalanche_pdf_size_d_0 = sum(mask_avalanche_d_0_bc)/timesteps
  norm_avalanche_pdf_size_s_1 = sum(mask_avalanche_s_1_bc)/width
  norm_avalanche_pdf_size_d_1 = sum(mask_avalanche_d_1_bc)/timesteps

  mean_avalanche_pdf_size = np.mean([norm_avalanche_pdf_size_s_0,\
                                    norm_avalanche_pdf_size_d_0,\
                                    norm_avalanche_pdf_size_s_1,\
                                    norm_avalanche_pdf_size_d_1])
  max_avalanche_pdf_size = np.max([norm_avalanche_pdf_size_s_0,\
                                   norm_avalanche_pdf_size_d_0,\
                                   norm_avalanche_pdf_size_s_1,\
                                   norm_avalanche_pdf_size_d_1])

  return np.tanh(5*(0.9*max_avalanche_pdf_size+0.1*mean_avalanche_pdf_size))

def calculate_data_score(data):
  fit = powerlaw.Fit(data, xmin =1, discrete= True)
  alpha = fit.power_law.alpha
  ksdist = fit.power_law.D
  R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
  R_exp = R_exp if p_exp < 0.1 else 0

  R = R_exp

  return alpha, ksdist, R

def sigmoid(x, smooth=0.01):
  return 1. / (1. + np.exp(-x*smooth))

def norm_comparison_ratio(R_list):
  return sigmoid(0.9*np.max(R_list) + 0.1*np.mean(R_list))

def calculate_comparison_ratio(data):
  fit = powerlaw.Fit(data, xmin =1, discrete= True)
  R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
  R = R_exp if p_exp < 0.1 else 0

  return R


def evaluate_result(ca_result, filename=None):
  avalanche_s_0 = getarray_avalanche_size(ca_result, 0)
  print(avalanche_s_0[:10])
  print(avalanche_s_0[-10:])
  avalanche_d_0 = getarray_avalanche_duration(ca_result, 0)
  avalanche_s_0_bc = np.bincount(avalanche_s_0)[1:] if len(avalanche_s_0) > 5 else []
  avalanche_d_0_bc = np.bincount(avalanche_d_0)[1:] if len(avalanche_d_0) > 5 else []

  avalanche_s_1 = getarray_avalanche_size(ca_result, 1)
  print(avalanche_s_1[:10])
  print(avalanche_s_1[-10:])
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
  norm_avalanche_pdf_size = 0
  norm_linscore_res = 0
  norm_ksdist_res = 0
  norm_coef_res = 0
  norm_unique_states = 0
  norm_R_res = 0

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

    R_list = []
    R_list.append(calculate_comparison_ratio(avalanche_s_0))
    R_list.append(calculate_comparison_ratio(avalanche_d_0))
    R_list.append(calculate_comparison_ratio(avalanche_s_1))
    R_list.append(calculate_comparison_ratio(avalanche_d_1))

    norm_avalanche_pdf_size = normalize_avalanche_pdf_size(mask_avalanche_s_0_bc,\
                                                           mask_avalanche_d_0_bc,\
                                                           mask_avalanche_s_1_bc,\
                                                           mask_avalanche_d_1_bc)

    print("linscore_list", linscore_list)
    print("coef_list", coef_list)
    print("ksdist_list", ksdist_list)

    norm_linscore_res = norm_linscore(linscore_list)
    norm_ksdist_res = norm_ksdist(ksdist_list)
    norm_coef_res = norm_coef(coef_list)
    norm_unique_states = ((np.unique(ca_result, axis=0).shape[0]) / ca_result.shape[1])

    print("norm_avalanche_pdf_size", norm_avalanche_pdf_size)
    print("norm_linscore_res", norm_linscore_res)
    print("norm_ksdist_res", norm_ksdist_res)
    print("norm_coef_res", norm_coef_res)
    print("norm_unique_states", norm_unique_states)

    fitness = norm_ksdist_res**2 + norm_unique_states + norm_avalanche_pdf_size + norm_linscore_res**2

    if fitness > 3.5:
      R_list = []
      R_list.append(calculate_comparison_ratio(avalanche_s_0))
      R_list.append(calculate_comparison_ratio(avalanche_d_0))
      R_list.append(calculate_comparison_ratio(avalanche_s_1))
      R_list.append(calculate_comparison_ratio(avalanche_d_1))
      print("R_list", R_list)
      norm_R_res = norm_comparison_ratio(R_list)
      print("norm_R_res", norm_R_res)
      fitness = fitness + norm_R_res

  val_dict = {}
  val_dict["norm_ksdist_res"] = norm_ksdist_res
  val_dict["norm_coef_res"] = norm_coef_res
  val_dict["norm_unique_states"] = norm_unique_states
  val_dict["norm_avalanche_pdf_size"] = norm_avalanche_pdf_size
  val_dict["norm_linscore_res"] = norm_linscore_res
  val_dict["norm_R_res"] = norm_R_res
  val_dict["fitness"] = fitness

  print("Fitness", fitness)
  return fitness, val_dict

def plot_ca_result(ca_result):
  ca_result_img = ca_result.astype(np.uint8)*255

  img = Image.fromarray(ca_result_img).resize((5*width,5*timesteps), Image.NEAREST)
  timestr = time.strftime("%Y%m%d-%H%M%S")
  img.save("results/evolved_stochastic_ca_criticality_"+timestr+".png")

def plot_distribution(distribution, fitobj=None, title=""):
  plt.figure()
  pdf = distribution/sum(distribution)
  pdf[pdf == 0] = np.nan
  x = np.linspace(1,len(pdf),len(pdf))
  plt.loglog(x, pdf, "-", label="pdf")

  if fitobj == None:
    plt.loglog(x, [pdf[0]*xx**(-2.) for xx in x], label="slope=-2", linestyle = "--")
  else:
    plt.loglog(x, np.power(10,fitobj.predict(np.log10(x).reshape(-1,1))), linestyle = "--", label="slope={0:.2f}".format(fitobj.coef_[0]))

  plt.legend(loc='upper right')
  plt.title(title)
  plt.show()

avalanche_s_0 = getarray_avalanche_size(ca_result, 0)
avalanche_d_0 = getarray_avalanche_duration(ca_result, 0)
avalanche_s_0_bc = np.bincount(avalanche_s_0)[1:]
avalanche_d_0_bc = np.bincount(avalanche_d_0)[1:]

avalanche_s_1 = getarray_avalanche_size(ca_result, 1)
avalanche_d_1 = getarray_avalanche_duration(ca_result, 1)
avalanche_s_1_bc = np.bincount(avalanche_s_1)[1:]
avalanche_d_1_bc = np.bincount(avalanche_d_1)[1:]

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


powerlaw_stats(avalanche_s_0, "avalanche_s_0_")
powerlaw_stats(avalanche_d_0, "avalanche_d_0_")
powerlaw_stats(avalanche_s_1, "avalanche_s_1_")
powerlaw_stats(avalanche_d_1, "avalanche_d_1_")

evaluate_result(ca_result)
plot_ca_result(ca_result)

print("Total time:", time.time()-start)

np.savez("dict_avalanche_bincount.npz", avalanche_s_0_bc, avalanche_d_0_bc, avalanche_s_1_bc, avalanche_d_1_bc)
