""""
    The interface of the Evolution Algorithm Lib Geatpy.
"""

import logging
import numpy as np
import geatpy as ea
from ea_libs.ea_problem import CompressionProblem

msglogger = logging.getLogger()


class ArgsContainer(object):
    def __init__(self):
        pass


class EALibInterface(object):
    """Interface to the Geatpy implementation."""
    def ea_algorithm_factory(self, algorithm_name):
        return {
            "soea_DE_best_1_L_templet": ea.soea_DE_best_1_L_templet
        }[algorithm_name]

    def solve(self, env, args):
        msglogger.info("EAC: Using Geatpy.")
        """====Instanciate the Problem object===="""
        problem = CompressionProblem(env, args)
        """====Set up the Population===="""
        Encoding = args.ea_encoding_type
        NIND = args.ea_population_size
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
                          problem.borders)
        population = ea.Population(Encoding, Field, NIND)
        """"====Setup the arguments of the Evolution Algorithm.===="""
        self.ea_algorithm_fn = self.ea_algorithm_factory(args.ea_algorithm)
        myAlgorithm = self.ea_algorithm_fn(problem, population)
        myAlgorithm.MAXGEN = args.ea_max_gen
        myAlgorithm.mutOper.F = args.ea_mutOper_F
        myAlgorithm.recOper.XOVR = args.ea_recOper_XOVR
        myAlgorithm.drawing = args.ea_drawing_type
        """====Call the algorithm template===="""
        [poptlation, obj_trace, var_trace] = myAlgorithm.run()
        """====Output the results===="""
        best_gen = np.argmax(obj_trace[:, 1])
        best_ObjV = obj_trace[best_gen, 1]

        # msglogger.info("\tThe best Objection value is: %s" % best_ObjV)
        # msglogger.info("\tThe best compression schedule is: ")
        # for i in range(var_trace.shape[1]):
        #     msglogger.info("state_id : %d compress_ratio : %s" %
        #                    (args.state_id, var_trace[best_gen, i]))
        msglogger.info("\tValid evolution generations: %s" %
                       obj_trace.shape[0])
        # msglogger.info("\tBest generation: %s" % best_gen + 1)
        msglogger.info("\tEvaluation iterations:%s" % myAlgorithm.evalsNum)
        msglogger.info("\tRun time: %s" % myAlgorithm.passTime)
