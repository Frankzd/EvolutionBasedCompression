"""
Define the PROBLEM for  Evolution Algorithm.
"""
import logging
import random
from typing import List

import geatpy as ea
import numpy as np

from environment import Environment

msglogger = logging.getLogger()


class CompressionProblem(ea.Problem):
  """ Define the CompressionProblem
	* targets is the optimization targets, options are accuracy,
		macs , params and time,meanwhile only support accuracy.
	* compress_layers are the layers that can be pruned, which in 
		this case determines the number of Dims.
	* lb is the lower bound of x
	* ub is the upper bound of x
	* args
  """

  def __init__(self, env: Environment, args: dict) -> None:
    name = 'MyProblem'
    # todo current version only support for the optimization of accuracy.
    M = 1  # maximize accuracy
    maxormins = [-1]
    Dim = env.compression_action_dim
    self.env = env
    varTypes = [0] * Dim
    self.lb, self.ub = args.ea_action_range
    lb_list = [self.lb] * Dim
    ub_list = [self.ub] * Dim
    lbin = [0] * Dim
    ubin = [1] * Dim
    self.target_density = args.ea_target_density
    ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb_list,
                        ub_list, lbin, ubin)

  def Chrom_init(self, pop: ea.Population):
    """
				Initialize the Population due to target-density
		"""
    param_num_list = self.env.paramNumDistribution
    total_param = self.env.net_wrapper.total_nnz
    Chrom = pop.Chrom
    group_size = Chrom.shape[0]
    x_dim = Chrom.shape[1]
    index = [n for n in range(0, x_dim)]
    for i in range(group_size):
      random.shuffle(index)
      to_prune = round(total_param * (1 - self.target_density))
      total_left = sum(param_num_list)
      for j in range(x_dim):
        total_left = total_left - param_num_list[index[j]]
        if param_num_list[index[j]] * self.lb + total_left * self.ub < to_prune:
          min_prune_ratio = (to_prune -
                             total_left * self.ub) / param_num_list[index[j]]
          prune_ratio = self.ub if min_prune_ratio >= self.ub else random.uniform(
              min_prune_ratio, self.ub)
        else:
          prune_ratio = random.uniform(self.lb, self.ub)
        layer_pruned = prune_ratio * param_num_list[index[j]]
        to_prune = to_prune - layer_pruned
        Chrom[i,index[j]] = prune_ratio
      # print(Chrom[i,:])
      # print(sum(Chrom[i,:]*param_num_list)/total_param)
    return Chrom

  def aimFunc(self, pop: ea.Population):
    """ 
				Aim function, pop is the population that we send in
		"""
    Vars = pop.Phen
    func_value = np.empty([Vars.shape[0], self.M], float)
    # varsperiter = [[] for i in range(self.ngpus)]
    for i in range(np.size(Vars, 0)):
      func_value[i, :], _ = self.env.fast_eval(Vars[i, :])
      self.env.reset()

    self.env.next_generation()
    msglogger.info("\t At generation %d the best value is %.3f" %
                   (self.env.generation, func_value.max(axis=0)))
    pop.ObjV = func_value

    # Constrains
    param_num_list = np.array(self.env.paramNumDistribution)
    np.expand_dims(param_num_list, axis=1)
    target_prune_params = (
        1 - self.env.ea_cfg.target_density) * self.env.net_wrapper.total_nnz

    pop.CV = np.expand_dims(np.array(target_prune_params -
                                     np.dot(Vars, param_num_list)),
                            axis=1)
    # mask_matrix = np.empty(pop.CV.shape)
    # mask_matrix[pop.CV <= 0] = 1
    # mask_matrix[pop.CV > 0] = 0
    # pop.ObjV = func_value * mask_matrix
    # print(pop.ObjV)
    # print(pop.CV)


if __name__ == '__main__':
  targets1 = ['accuracy', 'macs', 'params', 'time']
  targets2 = ['accuracy']
  targets3 = ['accuracy', 'fuck']
