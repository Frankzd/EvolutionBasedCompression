""" 
    The environment wrapper of the model.
    As we get the action from the optimizer and apply it to the 
    environment.
    The environment will then return the reward of this action and 
    send it to the optimizer.
"""
import numpy as np
import copy
import os
import math
import logging
import gym
import torch
import distiller
from utils.net_wrapper import NetworkWrapper
from collections import OrderedDict, namedtuple
from utils.features_collector import collect_intermediate_featuremap_samples
from utils.ac_loggers import AMCStatsLogger, FineTuneStatsLogger

msglogger = logging.getLogger()


def log_ea_config(ea_cfg):
    try:
        msglogger.info('EA configuration:')
        for k, v in ea_cfg.items():
            msglogger.info("\t{} : {}".format(k, v))
    except TypeError as e:
        pass


class Environment(gym.Env):
    def __init__(self, model, app_args, ea_cfg, services):
        self.pylogger = distiller.data_loggers.PythonLogger(
            logging.getLogger("summaries"))
        logdir = logging.getLogger().logdir
        self.tflogger = distiller.data_loggers.TensorBoardLogger(logdir)
        self._render = False
        self.orig_model = copy.deepcopy(model)
        self.app_args = app_args
        self.ea_cfg = ea_cfg
        self.services = services
        self.best_val_1 = float('-inf')
        self.best_val_2 = float('-inf')
        self.best_action_list_val_1 = []
        self.best_action_list_val_2 = []

        try:
            modules_list = ea_cfg.modules_dict[app_args.arch]
        except KeyError:
            raise ValueError(
                "The config file does not specify the modules to compress for %s"
                % app_args.arch)
        self.net_wrapper = NetworkWrapper(model, app_args, services,
                                          modules_list, ea_cfg.pruning_pattern)
        self.original_model_macs, self.original_model_size = self.net_wrapper.get_resources_requirements(
        )
        self.generation = 0
        self.reset(init_only=True)
        # self._max_episode_steps = self.net_wrapper.model_metadata.num_pruned_layers()
        self.episode = 0
        # self.best_reward = float('-inf')
        self.action_low, self.action_high = ea_cfg.action_range
        self._log_model_info()
        log_ea_config(ea_cfg)
        self.stats_logger = AMCStatsLogger(os.path.join(logdir, 'amc.csv'))
        self.ft_stats_logger = FineTuneStatsLogger(
            os.path.join(logdir, 'ft_top1.csv'))
        if self.ea_cfg.pruning_method == "fm-reconstruction":
            self._collect_fm_reconstruction_samples(modules_list)

    def _collect_fm_reconstruction_samples(self, modules_list):
        """Run the forward-pass on the selected dataset and collect feature-map samples.

        These data will be used when we optimize the compressed-net's weights by trying
        to reconstruct these samples.
        """
        from functools import partial
        if self.ea_cfg.pruning_pattern != "channels":
            raise ValueError(
                "Feature-map reconstruction is only supported when pruning weights channels"
            )

        def acceptance_criterion(m, mod_names):
            # Collect feature-maps only for Conv2d layers, if they are in our modules list.
            return isinstance(
                m, (torch.nn.Conv2d,
                    torch.nn.Linear)) and m.distiller_name in mod_names

        # For feature-map reconstruction we need to collect a representative set
        # of inter-layer feature-maps
        from distiller.pruning import FMReconstructionChannelPruner
        collect_intermediate_featuremap_samples(
            self.net_wrapper.model, self.net_wrapper.validate,
            partial(acceptance_criterion, mod_names=modules_list),
            partial(FMReconstructionChannelPruner.cache_featuremaps_fwd_hook,
                    n_points_per_fm=self.ea_cfg.n_points_per_fm))

    @property
    def macsDistribution(self):
        """Return the macs per layer Distribution"""
        # distribution = []
        per_layer_macs = []
        # total_macs = self.net_wrapper.total_macs
        layer_num = self.net_wrapper.num_pruned_layers()
        for i in range(layer_num):
            layer_id = self.net_wrapper.model_metadata.pruned_idxs[i]
            layer = self.net_wrapper.get_pruned_layer(layer_id)
            per_layer_macs.append(self.net_wrapper.layer_net_macs(layer))
            # distribution.append(self.net_wrapper.layer_net_macs(layer)/total_macs)
        # print("distribution: %r"%distribution)
        # print("per_layer_macs:%r"%per_layer_macs)
        # print("total_macs is:%r"%total_macs)
        # print("sum of per_layer_macs is:%r"%(sum(per_layer_macs)))
        # print("sum of distribution is:%r"%(sum(distribution)))

        # exit(1)
        return per_layer_macs

    @property
    def compression_action_dim(self):
        """ Return the number of the layers to prune which is also the dimension of the action"""
        return self.net_wrapper.model_metadata.num_pruned_layers()

    def record_best_model(self,
                          generation_number,
                          episode_number,
                          best_reward,
                          is_val_1=True):
        """ Record the best model of a given generation among all episodes(population size)"""
        if is_val_1:
            msglogger.info(
                "Record the best current model of Generation %d Episode %d with the best reward %.3f"
                % (generation_number, episode_number, best_reward))

            self.best_model_val_1 = copy.deepcopy(self.net_wrapper.model)
        else:
            msglogger.info(
                "Record the best generation with the best reward %.3f" %
                best_reward)
            self.best_model_val_2 = copy.deepcopy(self.net_wrapper.model)

    def reset(self, init_only=False):
        """Reset the environment.
        This is invoked by the Agent.
        """
        # ? There are two ways to compress the network:
        # ! First way is to update the layer one by one,this is the current
        # ! method
        # todo second way is to update all the layers at once

        msglogger.info(
            "Resetting the environment (init_only={})".format(init_only))
        self.prev_action = 0
        self.model = copy.deepcopy(self.orig_model)
        if hasattr(self.net_wrapper.model, 'intermediate_fms'):
            self.model.intermediate_fms = self.net_wrapper.model.intermediate_fms
        self.net_wrapper.reset(self.model)
        self.removed_macs = 0
        self.action_history = []
        if init_only:
            return
        # return initial_observation

    def _log_model_info(self):
        msglogger.debug("Model %s has %d modules (%d pruned)",
                        self.app_args.arch,
                        self.net_wrapper.model_metadata.num_layers(),
                        self.net_wrapper.model_metadata.num_pruned_layers())
        msglogger.debug("\tTotal MACs: %s" %
                        distiller.pretty_int(self.original_model_macs))
        msglogger.debug("\tTotal weights:%s" %
                        distiller.pretty_int(self.original_model_size))


    @property
    def removed_macs_pct(self):
        """Return the amount of MACs removed so far.
        This is normalized to range 0..1
        """
        return self.removed_macs / self.original_model_macs

    @property
    def steps_per_episode(self):
        return self.net_wrapper.model_metadata.num_pruned_layers()

    def render(self, mode='human'):
        """Provide some feedback to the user about what's going on.
        This is invoked by the Agent.
        """
        if not self._render:
            return
        msglogger.info("Render Environment: current_episode=%d" % self.episode)
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

    def fast_eval(self,pruning_action_list):
        """
            Fast evaluation of the proposed pruning list using element-wise pruning method
        """
        msglogger.info("+" + "-" * 50 + "+")
        msglogger.info("Generation %d Episode %d is starting" %
                       (self.generation, self.episode))

        pruning_action_list = np.clip(pruning_action_list, self.action_low,
                                      self.action_high)
        msglogger.debug("\tAgent clipped pruning_action_list={}".format(
            pruning_action_list))

        # Calculate the final compression rate.
        total_macs_before, _ = self.net_wrapper.get_resources_requirements()
        layer_macs_before_list = []
        for state_id in range(len(pruning_action_list)):
            layer_id = self.net_wrapper.model_metadata.pruned_idxs[state_id]
            layer = self.net_wrapper.get_pruned_layer(layer_id)
            layer_macs_before_action = self.net_wrapper.layer_macs(layer)
            layer_macs_before_list.append(layer_macs_before_action)

        pruning_action_list = self.net_wrapper.remove_elements_list(
            fraction_to_prune_list=pruning_action_list.tolist(),
            prune_how=self.ea_cfg.pruning_method,
            group_size=self.ea_cfg.group_size,
            ranking_noise=self.ea_cfg.ranking_noise)

        total_macs_after_act, total_nnz_after_act = self.net_wrapper.get_resources_requirements(
        )
        # msglogger.info("total_macs_after_act %d total_nnz_after_act %d " %
        #                (total_macs_after_act, total_nnz_after_act))
        total_macs_after_list = []
        # Update the various counters after taking the step.
        self.removed_macs = (total_macs_before - total_macs_after_act)
        msglogger.debug("\tactual_action={}".format(pruning_action_list))
        msglogger.debug("\tself._removed_macs={}".format(self.removed_macs))
        msglogger.debug("\tself._removed_pct={}".format(self.removed_macs_pct))
        for state_id in range(len(pruning_action_list)):
            layer_id = self.net_wrapper.model_metadata.pruned_idxs[state_id]
            layer = self.net_wrapper.get_pruned_layer(layer_id)
            layer_macs_after_action = self.net_wrapper.layer_macs(layer)
            total_macs_after_list.append(layer_macs_after_action)
            msglogger.debug("\tlayer_macs_after_action: %.3f" %
                            layer_macs_after_action)
            msglogger.debug("\tlayer_macs_before_list[%d]: %.3f" %
                            (state_id, layer_macs_before_list[state_id]))
            msglogger.debug("\tpruning_action_list[%d]: %.3f" %
                            (state_id, pruning_action_list[state_id]))
            msglogger.debug(
                "\tDevide: %.3f" %
                (layer_macs_after_action / layer_macs_before_list[state_id]))
            msglogger.debug("\tbaseline: %.3f" %
                            (1 - pruning_action_list[state_id]))
            # assert math.isclose(
            #     layer_macs_after_action / layer_macs_before_list[state_id],
            #     1 - pruning_action_list[state_id])

        self.action_history = pruning_action_list
        msglogger.info("Generation %d Episode %d is ending" %
                       (self.generation, self.episode))
        reward, top1, top5, vloss = self.compute_reward(
            total_macs_after_act, total_nnz_after_act)
        self.finalize_episode(reward, (top1, top5, vloss),
                              total_macs_after_act, total_nnz_after_act,
                              self.action_history)
        normalized_macs = total_macs_after_act / self.original_model_macs * 100
        info = {"accuracy": top1, "compress_ratio": normalized_macs}
        if self.ea_cfg.protocol == "mac-constrained":
            # Sanity check (special case only for "mac-constrained")
            # assert self.removed_macs_pct >= 1 - self.ea_cfg.target_density - 0.002
            pass

        self.episode += 1
        return reward, info

    def step(self, pruning_action_list):
        """Take a step, given an action list.
        The action list represents the desired sparsity for the all the prunable layers 
            (i.e. the percentage of weights to remove)
        This function is invoked by the agent.
        """
        msglogger.info("+" + "-" * 50 + "+")
        msglogger.info("Generation %d Episode %d is starting" %
                       (self.generation, self.episode))

        pruning_action_list = np.clip(pruning_action_list, self.action_low,
                                      self.action_high)
        msglogger.debug("\tAgent clipped pruning_action_list={}".format(
            pruning_action_list))

        # Calculate the final compression rate.
        total_macs_before, _ = self.net_wrapper.get_resources_requirements()
        msglogger.info(
                "\033[1;31m Total_macs_before=%d !\033[0m"%total_macs_before)
        layer_macs_before_list = []
        for state_id in range(len(pruning_action_list)):
            layer_id = self.net_wrapper.model_metadata.pruned_idxs[state_id]
            layer = self.net_wrapper.get_pruned_layer(layer_id)
            layer_macs_before_action = self.net_wrapper.layer_macs(layer)
            layer_macs_before_list.append(layer_macs_before_action)

        pruning_action_list = self.net_wrapper.remove_structures_list(
            fraction_to_prune_list=pruning_action_list.tolist(),
            prune_what=self.ea_cfg.pruning_pattern,
            prune_how=self.ea_cfg.pruning_method,
            group_size=self.ea_cfg.group_size,
            ranking_noise=self.ea_cfg.ranking_noise)

        total_macs_after_act, total_nnz_after_act = self.net_wrapper.get_resources_requirements(
        )

        total_macs_after_list = []
        # Update the various counters after taking the step.
        self.removed_macs = (total_macs_before - total_macs_after_act)
        msglogger.debug("\tactual_action={}".format(pruning_action_list))
        msglogger.debug("\tself._removed_macs={}".format(self.removed_macs))
        msglogger.debug("\tself._removed_pct={}".format(self.removed_macs_pct))
        for state_id in range(len(pruning_action_list)):
            layer_id = self.net_wrapper.model_metadata.pruned_idxs[state_id]
            layer = self.net_wrapper.get_pruned_layer(layer_id)
            layer_macs_after_action = self.net_wrapper.layer_macs(layer)
            total_macs_after_list.append(layer_macs_after_action)
            msglogger.debug("\tlayer_macs_after_action: %.3f" %
                            layer_macs_after_action)
            msglogger.debug("\tlayer_macs_before_list[%d]: %.3f" %
                            (state_id, layer_macs_before_list[state_id]))
            msglogger.debug("\tpruning_action_list[%d]: %.3f" %
                            (state_id, pruning_action_list[state_id]))
            msglogger.debug(
                "\tDevide: %.3f" %
                (layer_macs_after_action / layer_macs_before_list[state_id]))
            msglogger.debug("\tbaseline: %.3f" %
                            (1 - pruning_action_list[state_id]))
            # assert math.isclose(
            #     layer_macs_after_action / layer_macs_before_list[state_id],
            #     1 - pruning_action_list[state_id])

        self.action_history = pruning_action_list
        msglogger.info("Generation %d Episode %d is ending" %
                       (self.generation, self.episode))
        reward, top1, top5, vloss = self.compute_reward(
            total_macs_after_act, total_nnz_after_act)
        self.finalize_episode(reward, (top1, top5, vloss),
                              total_macs_after_act, total_nnz_after_act,
                              self.action_history)
        normalized_macs = total_macs_after_act / self.original_model_macs * 100
        info = {"accuracy": top1, "compress_ratio": normalized_macs}
        if self.ea_cfg.protocol == "mac-constrained":
            # Sanity check (special case only for "mac-constrained")
            # assert self.removed_macs_pct >= 1 - self.ea_cfg.target_density - 0.002
            pass

        self.episode += 1
        return reward, info

    def next_generation(self):
        """Deal with the next generation:
            1. generation_number++
            2. find the best model in validation set 2
        """

        self.net_wrapper.reset(self.best_model_val_1)
        #! debug check if reward is equal on validation set 1
        top1, top5, vloss = self.net_wrapper.validate()
        reward_validation_set_1 = self.ea_cfg.reward_fn(
            self, top1, top5, vloss, self.best_total_macs_val_1)
        print("reward validation set 1 is %.6f" % reward_validation_set_1)
        print("self.best_val_1 is %.6f" % self.best_val_1)
        # assert math.isclose(reward_validation_set_1, self.best_val_1)
        #! Upper code is used for debugging

        top1, top5, vloss = self.net_wrapper.validate(False)
        reward_validation_set_2 = self.ea_cfg.reward_fn(
            self, top1, top5, vloss, self.best_total_macs_val_1)
        msglogger.info("The reward on Validation set 2 is %.3f" %
                       reward_validation_set_2)
        self.finalize_generation(reward_validation_set_2, (top1, top5, vloss),
                                 self.best_total_macs_val_1,
                                 self.best_total_nnz_val_1,
                                 self.best_action_list_val_1)
        self.episode = 0
        self.generation += 1

    def run_test(self):
        """Run the final test on test set"""
        self.net_wrapper.reset(self.best_model_val_2)
        top1, top5, vloss = self.net_wrapper.test()
        reward_test_set = self.ea_cfg.reward_fn(self, top1, top5, vloss,
                                                self.best_total_macs_val_2)
        normalized_macs = self.best_total_macs_val_2 / self.original_model_macs * 100
        msglogger.info("The final reward on test set is %.3f" %
                       reward_test_set)
        msglogger.info("The compression ratio of the best model is %.3f" %
                       normalized_macs)
        msglogger.info("The final action is %s" % self.best_action_list_val_2)

    def finalize_episode(self,
                         reward,
                         val_results,
                         total_macs,
                         total_nnz,
                         action_history,
                         log_stats=True):
        """Write the details of one network to the logger and create a checkpoint file"""
        top1, top5, vloss = val_results
        normalized_macs = total_macs / self.original_model_macs * 100
        normalized_nnz = total_nnz / self.original_model_size * 100

        if reward > self.best_val_1:
            self.best_val_1 = reward
            self.record_best_model(self.generation, self.episode, reward)
            self.best_total_macs_val_1 = total_macs
            self.best_total_nnz_val_1 = total_nnz
            self.best_action_list_val_1 = action_history
            msglogger.info(
                "Best reward={} generation={} episode={} top1={} top5={}".
                format(reward, self.generation, self.episode, top1, top5))

        import json
        performance = self.net_wrapper.performance_summary()
        fields = [
            self.episode, top1, reward, total_macs, normalized_macs,
            normalized_nnz,
            json.dumps(action_history),
            json.dumps(performance)
        ]
        self.stats_logger.add_record(fields)
        msglogger.info(
            "Top1:%.2f - compute: %.2f%% - params:%.2f%% - actions:%s", top1,
            normalized_macs, normalized_nnz, self.action_history)
        if log_stats:
            stats = ('Performance/EpisodeEnd/',
                     OrderedDict([('Loss', vloss), ('Top1', top1),
                                  ('Top5', top5), ('reward', reward),
                                  ('total_macs', int(total_macs)),
                                  ('macs_normalized', normalized_macs),
                                  ('log(total_macs)', math.log(total_macs)),
                                  ('total_nnz', int(total_nnz))]))
            distiller.log_training_progress(
                stats,
                None,
                self.episode,
                steps_completed=0,
                total_steps=1,
                log_freq=1,
                loggers=[self.tflogger, self.pylogger])

    def finalize_generation(self,
                            reward,
                            val_results,
                            total_macs,
                            total_nnz,
                            action_history,
                            log_stats=True):
        """Write the details of one network to the logger and create a checkpoint file"""
        top1, top5, vloss = val_results
        normalized_macs = total_macs / self.original_model_macs * 100
        normalized_nnz = total_nnz / self.original_model_size * 100

        if reward > self.best_val_2:
            self.best_val_2 = reward
            self.record_best_model(self.generation, 0, reward, False)
            self.best_total_macs_val_2 = total_macs
            self.best_total_nnz_val_2 = total_nnz
            self.best_action_list_val_2 = action_history
            ckpt_name = self.save_checkpoint(is_best=True)
            msglogger.info(
                "\033[1;31m New best result on Validation set 2 !\033[0m")
            msglogger.info("\033[1;31m Generation: %d Reward: %.3f\033[0m" %
                           (self.generation, reward))
            msglogger.info(
                "Best reward={} generation={} top1={} top5={}".format(
                    reward, self.generation, top1, top5))
        else:
            ckpt_name = self.save_checkpoint(is_best=False)

        import json
        performance = self.net_wrapper.performance_summary()
        fields = [
            self.generation, top1, reward, total_macs, normalized_macs,
            normalized_nnz, ckpt_name,
            json.dumps(action_history),
            json.dumps(performance)
        ]
        self.stats_logger.add_record(fields)
        msglogger.info(
            "Top1:%.2f - compute: %.2f%% - params:%.2f%% - actions:%s", top1,
            normalized_macs, normalized_nnz, action_history)
        if log_stats:
            stats = ('Performance/GenerationEnd/',
                     OrderedDict([('Loss', vloss), ('Top1', top1),
                                  ('Top5', top5), ('reward', reward),
                                  ('total_macs', int(total_macs)),
                                  ('macs_normalized', normalized_macs),
                                  ('log(total_macs)', math.log(total_macs)),
                                  ('total_nnz', int(total_nnz))]))
            distiller.log_training_progress(
                stats,
                None,
                self.generation,
                steps_completed=0,
                total_steps=1,
                log_freq=1,
                loggers=[self.tflogger, self.pylogger])
        # reset the best_val_1 on validation set per generation
        self.best_val_1 = float('-inf')
        self.best_total_macs_val_1 = 0.0
        self.best_total_nnz_val_1 = 0.0

    def save_checkpoint(self, is_best=False):
        """Save the learned-model checkpoint"""
        generation = str(self.generation).zfill(3)
        if is_best:
            fname = "BEST_ea_generation_{}".format(generation)
        else:
            fname = "adc_episode_{}".format(generation)
        if is_best or self.ea_cfg.save_chkpts:
            # Always save the best episodes, and depending on amc_cfg.save_chkpts save all other episodes
            scheduler = self.net_wrapper.create_scheduler()
            extras = {"creation_masks": self.net_wrapper.sparsification_masks}
            self.services.save_checkpoint_fn(epoch=0,
                                             model=self.net_wrapper.model,
                                             scheduler=scheduler,
                                             name=fname,
                                             extras=extras)
            del scheduler
        return fname

    def compute_reward(self, total_macs, total_nnz):
        """Compute the reward.

        We use the validation set(the size of validation set is 
        configured when the data-loader is instantiated)
        """
        num_elements = distiller.model_params_size(self.model,
                                                   param_dims=[2, 4],
                                                   param_types=['weight'])

        # Fine tune (This is a nop if self.ea_cfg.num_ft_epochs==0)
        accuracies = self.net_wrapper.train(self.ea_cfg.num_ft_epochs,
                                            self.episode)
        self.ft_stats_logger.add_record([self.episode, accuracies])

        top1, top5, vloss = self.net_wrapper.validate()
        reward = self.ea_cfg.reward_fn(self, top1, top5, vloss, total_macs)
        return reward, top1, top5, vloss