""" Evolve rules of CA 1D """

import numpy as np
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
from evodynamic.evolution import ga
import time

width = 1000
timesteps = 1000

# Fitness function. Number of white and black cells must be equal
def evaluate_result(ca_result):
  fitness = 1.-2.*abs(np.mean(ca_result)-0.5) # 1 = best score, 0 = worst score
  val_dict = {}
  val_dict["fitness"] = fitness
  return fitness, val_dict

# genome is a list of integers between 0 and 255
def evaluate_genome(genome=[110], show_result = False):
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
  print("Execution time:", time.time()-start)
  exp.close()
  fitness, val_dict = evaluate_result(exp.get_monitor("g_ca", "g_ca_bin"))

  return fitness, val_dict

start_total = time.time()
best_genome = ga.evolve_rules(evaluate_genome, pop_size=10, generation=10)

print("TOTAL Execution time:", time.time()-start_total)
print("Best genome", best_genome)
print("Final fitness", evaluate_genome(best_genome)[0])