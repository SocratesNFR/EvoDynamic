""" Cellular automata 1D - Self-organized criticality"""

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import numpy as np
from sklearn.linear_model import LinearRegression
import time

width = 1000
timesteps = 1000

exp = experiment.Experiment()
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin')
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx,
                                           is_wrapped_ca=True)

#g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
#                            activation_func=act.rule_binary_ca_1d_width3_func,\
#                             fargs=(110,))

#g_ca.add_internal_connection(state_name='g_ca_bin', connection=g_ca_bin_conn,\
#                             activation_func=act.rule_binary_ca_1d_width3_func,\
#                             fargs=(10,))

fargs_list = [([0.8653582694285038, 0.07574324505979713, 0.0, 0.6701419392338681,\
                0.02243120000370638, 0.46365825379340864, 0.7473194740998363, 1.0],)]

#fargs_list = [([0.8844229523381606, 0.16105272036927262, 0.531159153794522,\
#                0.10896772965937962, 0.03419006259328916, 0.8514302121431078,\
#                0.993618976607495, 0.9986691476433076],)]

#[0.9434366987096601, 0.0, 0.2664803614620113, 1.0, 0.0, 0.5734834314810755, 0.004014194229735535, 0.6023044067207612]
#fargs_list = [([0.9434366987096601, 0.0, 0.2664803614620113, 1.0, 0.0,\
#                0.5734834314810755, 0.004014194229735535, 0.6023044067207612],)]

# Test 2
#fargs_list = [([0.4934885953923782, 0.4661613668731218, 1.0, 0.3138882790836618,\
#                0.0, 0.007657978203402652, 0.6245438662473278, 0.9949950700207485],)]


#[0.4073302786551547, 0.07588345236524413, 0.15676889915093514, 0.08356471216999127, 0.00879616388063742, 0.403847608690661, 0.6751633501344202, 0.8846790551722384]
#fargs_list = [([0.4073302786551547, 0.07588345236524413, 0.15676889915093514,\
#                0.08356471216999127, 0.00879616388063742, 0.403847608690661,\
#                0.6751633501344202, 0.8846790551722384],)]


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

def powerlaw_stats(data):
  import powerlaw
  fit = powerlaw.Fit(data, xmin =1, discrete= True)
  print()
  print("alpha", fit.power_law.alpha)
  print("xmin", fit.power_law.xmin)
  print("sigma", fit.power_law.sigma)
  print("KSdist", fit.power_law.D)
  print("fit.distribution_compare('power_law', 'exponential')", fit.distribution_compare('power_law', 'exponential'))
  print("fit.distribution_compare('power_law', 'truncated_power_law')", fit.distribution_compare('power_law', 'truncated_power_law'))
  print("fit.distribution_compare('power_law', 'lognormal')", fit.distribution_compare('power_law', 'lognormal'))
  print()

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

def getarray_avalanche_count(x, value):
  list_avalance_count = []
  if value in x:
    x0size, x1size = x.shape  
    for i in range(x0size):
      if value in x[i,:]:
        list_avalance_count.append(list(x[i,:]).count(value))
  return np.array(list_avalance_count)

def getarray_avalanche_duration(x, value):
  list_avalance_duration = []
  if value in x:
    x0size, x1size = x.shape
    for i in range(x1size):
      if value in x[:,i]:
        list_avalance_duration.extend(getdict_cluster_size(x[:,i])[value])
  return np.array(list_avalance_duration)

def plot_distribution(distribution, fitobj=None, title=""):
  import matplotlib.pyplot as plt
  plt.figure()
  pdf = distribution/sum(distribution)
  x = np.linspace(1,len(pdf),len(pdf))
  plt.loglog(x, pdf, "o-", label="pdf")

  if fitobj == None:
    plt.loglog(x, [pdf[0]*xx**(-5.) for xx in x], label="slope=-5")
  else:
    plt.loglog(x, np.power(10,fitobj.predict(np.log10(x).reshape(-1,1))), label="slope={0:.2f}".format(fitobj.coef_[0]))

  plt.legend(loc='upper right')
  plt.title(title)
  plt.show()

avalanche_s_0 = getarray_avalanche_size(ca_result, 0)
#avalanche_s_0 = avalanche_s_0[avalanche_s_0 < width*0.95]

avalanche_d_0 = getarray_avalanche_duration(ca_result, 0)
#avalanche_d_0 = avalanche_d_0[avalanche_d_0 < timesteps*0.95]

#avalanche_c_0 = getarray_avalanche_count(ca_result, 0)
avalanche_s_0_bc = np.bincount(avalanche_s_0)[1:]
avalanche_d_0_bc = np.bincount(avalanche_d_0)[1:]
#avalanche_c_0_bc = np.bincount(avalanche_c_0)[1:]

avalanche_s_1 = getarray_avalanche_size(ca_result, 1)
#avalanche_s_1 = avalanche_s_1[avalanche_s_1 < width*0.95]

avalanche_d_1 = getarray_avalanche_duration(ca_result, 1)
#avalanche_d_1 = avalanche_d_1[avalanche_d_1 < timesteps*0.95]

#avalanche_c_1 = getarray_avalanche_count(ca_result, 1)
avalanche_s_1_bc = np.bincount(avalanche_s_1)[1:]
avalanche_d_1_bc = np.bincount(avalanche_d_1)[1:]
#avalanche_c_1_bc = np.bincount(avalanche_c_1)[1:]

avalanche_s_0_bc = avalanche_s_0_bc/sum(avalanche_s_0_bc)
avalanche_d_0_bc = avalanche_d_0_bc/sum(avalanche_d_0_bc)
avalanche_s_1_bc = avalanche_s_1_bc/sum(avalanche_s_1_bc)
avalanche_d_1_bc = avalanche_d_1_bc/sum(avalanche_d_1_bc)


log_avalanche_s_0_bc = np.log10(avalanche_s_0_bc)
log_avalanche_d_0_bc = np.log10(avalanche_d_0_bc)
log_avalanche_s_1_bc = np.log10(avalanche_s_1_bc)
log_avalanche_d_1_bc = np.log10(avalanche_d_1_bc)

log_avalanche_s_0_bc = np.where(np.isfinite(log_avalanche_s_0_bc), log_avalanche_s_0_bc, 0)
log_avalanche_d_0_bc = np.where(np.isfinite(log_avalanche_d_0_bc), log_avalanche_d_0_bc, 0)
log_avalanche_s_1_bc = np.where(np.isfinite(log_avalanche_s_1_bc), log_avalanche_s_1_bc, 0)
log_avalanche_d_1_bc = np.where(np.isfinite(log_avalanche_d_1_bc), log_avalanche_d_1_bc, 0)

fit_avalanche_s_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_0_bc)+1)).reshape(-1,1), log_avalanche_s_0_bc, sample_weight=[1 if w<=10 else 0 for w in range(len(avalanche_s_0_bc))])
fit_avalanche_d_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_0_bc)+1)).reshape(-1,1), log_avalanche_d_0_bc, sample_weight=[1 if w<=10 else 0 for w in range(len(avalanche_d_0_bc))])
fit_avalanche_s_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_1_bc)+1)).reshape(-1,1), log_avalanche_s_1_bc, sample_weight=[1 if w<=10 else 0 for w in range(len(avalanche_s_1_bc))])
fit_avalanche_d_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_1_bc)+1)).reshape(-1,1), log_avalanche_d_1_bc, sample_weight=[1 if w<=10 else 0 for w in range(len(avalanche_d_1_bc))])

print(fit_avalanche_s_0_bc.score(np.log10(np.arange(1,len(avalanche_s_0_bc)+1)).reshape(-1,1), log_avalanche_s_0_bc))
print(fit_avalanche_d_0_bc.score(np.log10(np.arange(1,len(avalanche_d_0_bc)+1)).reshape(-1,1), log_avalanche_d_0_bc))
print(fit_avalanche_s_1_bc.score(np.log10(np.arange(1,len(avalanche_s_1_bc)+1)).reshape(-1,1), log_avalanche_s_1_bc))
print(fit_avalanche_d_1_bc.score(np.log10(np.arange(1,len(avalanche_d_1_bc)+1)).reshape(-1,1), log_avalanche_d_1_bc))
print(np.unique(ca_result, axis=0).shape[0], timesteps)
print(fit_avalanche_s_0_bc.coef_, fit_avalanche_d_0_bc.coef_, fit_avalanche_s_1_bc.coef_, fit_avalanche_d_1_bc.coef_)

print("KSdist")
print(KSdist(np.power(10,fit_avalanche_s_0_bc.predict(np.log10(np.arange(1,len(avalanche_s_0_bc)+1)).reshape(-1,1))), avalanche_s_0_bc))
print(KSdist(np.power(10,fit_avalanche_d_0_bc.predict(np.log10(np.arange(1,len(avalanche_d_0_bc)+1)).reshape(-1,1))), avalanche_d_0_bc))
print(KSdist(np.power(10,fit_avalanche_s_1_bc.predict(np.log10(np.arange(1,len(avalanche_s_1_bc)+1)).reshape(-1,1))), avalanche_s_1_bc))
print(KSdist(np.power(10,fit_avalanche_d_1_bc.predict(np.log10(np.arange(1,len(avalanche_d_1_bc)+1)).reshape(-1,1))), avalanche_d_1_bc))


plot_distribution(avalanche_s_0_bc, fit_avalanche_s_0_bc, "Avalanche size | Elementary CA rule 110+10 | v=0 | N=10^4 | t=10^5")
plot_distribution(avalanche_d_0_bc, fit_avalanche_d_0_bc, "Avalanche duration | Elementary CA rule 110+10 | v=0 | N=10^4 | t=10^5")

plot_distribution(avalanche_s_1_bc, fit_avalanche_s_1_bc, "Avalanche size | Elementary CA rule 110+10 | v=1 | N=10^4 | t=10^5")
plot_distribution(avalanche_d_1_bc, fit_avalanche_d_1_bc, "Avalanche duration | Elementary CA rule 110+10 | v=1 | N=10^4 | t=10^5")

powerlaw_stats(avalanche_s_0)
powerlaw_stats(avalanche_d_0)
powerlaw_stats(avalanche_s_1)
powerlaw_stats(avalanche_d_1)

np.savez("dict_avalanche_bincount.npz", avalanche_s_0_bc, avalanche_d_0_bc, avalanche_s_1_bc, avalanche_d_1_bc)
