"""
    Define the PROBLEM for  Evolution Algorithm.
"""
from typing import List
import numpy as np
import geatpy as ea
from environment import Environment
import logging

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
        #todo current version only support for the optimization of accuracy.
        M = 1  # maximize accuracy
        maxormins = [-1]
        Dim = env.compression_action_dim
        self.env = env
        varTypes = [0] * Dim
        lb, ub = args.ea_action_range
        lb_list = [lb] * Dim
        ub_list = [ub] * Dim
        lbin = [0] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb_list,
                            ub_list, lbin, ubin)

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
        # pop.ObjV = func_value

        #Constrains
        macs_per_layer = np.array(self.env.macsDistribution)
        np.expand_dims(macs_per_layer,axis=1)
        target_prune_macs = (1 - self.env.ea_cfg.target_density) * self.env.net_wrapper.total_macs
        
        pop.CV = np.expand_dims(np.array(target_prune_macs-np.dot(Vars,macs_per_layer)),axis=1)
        mask_matrix = np.empty(pop.CV.shape)
        mask_matrix[pop.CV<=0]=1
        mask_matrix[pop.CV>0]=0
        pop.ObjV = func_value * mask_matrix
        print(pop.ObjV)
        # print(pop.CV)

        


if __name__ == '__main__':
    targets1 = ['accuracy', 'macs', 'params', 'time']
    targets2 = ['accuracy']
    targets3 = ['accuracy', 'fuck']
