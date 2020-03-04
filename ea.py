"""
    The main program to run the Evolution Algorithm Compression
"""

import os
import logging
import distiller
from functools import partial
import distiller.apputils as apputils
from environment import Environment
from rewards import reward_factory
import distiller.apputils.image_classifier as classifier

msglogger = logging.getLogger()

#todo: 1. Add a second validation loader, to overcome the over-fit
#todo: 2. Complete the logger, print the generation and the episode
#todo: 3. Save the best checkpoint


class AutoCompressionSampleApp(classifier.ClassifierCompressor):
    def __init__(self, args, script_dir):
        super().__init__(args, script_dir)

    def train_auto_compressor(self):
        using_fm_reconstruction = self.args.ea_prune_method == 'fm-reconstruction'
        fixed_subset, sequential = (using_fm_reconstruction,
                                    using_fm_reconstruction)
        msglogger.info("AMC: fixed_subset=%s\tsequential=%s" %
                       (fixed_subset, sequential))

        # version1: use one validation set
        # train_loader, val_loader1, test_loader = classifier.load_data(
        #     self.args, fixed_subset, sequential)
        # val_loader2 = None

        train_loader, val_loader_list, test_loader = classifier.load_data(
            self.args, fixed_subset, sequential)

        self.args.display_confusion = False

        validation_fn_list = []
        for validation_idx in range(self.args.validation_num):
            validation_fn = partial(
                classifier.test,
                test_loader=val_loader_list[validation_idx],
                criterion=self.criterion,
                loggers=self.pylogger,
                args=self.args,
                activations_collectors=None)
            validation_fn_list.append(validation_fn)

        # validate_fn_1 = partial(classifier.test,
        #                         test_loader=val_loader1,
        #                         criterion=self.criterion,
        #                         loggers=self.pylogger,
        #                         args=self.args,
        #                         activations_collectors=None)
        train_fn = partial(classifier.train,
                           train_loader=train_loader,
                           criterion=self.criterion,
                           loggers=self.pylogger,
                           args=self.args)
        # validate_fn_2 = partial(classifier.test,
        #                         test_loader=val_loader2,
        #                         criterion=self.criterion,
        #                         loggers=self.pylogger,
        #                         args=self.args,
        #                         activations_collectors=None)

        test_fn = partial(classifier.test,
                          test_loader=test_loader,
                          criterion=self.criterion,
                          loggers=self.pylogger,
                          args=self.args,
                          activations_collectors=None)

        save_checkpoint_fn = partial(apputils.save_checkpoint,
                                     arch=self.args.arch,
                                     dir=msglogger.logdir)
        optimizer_data = {
            'lr': self.args.lr,
            'momentum': self.args.momentum,
            'weight_decay': self.args.weight_decay
        }
        return train_auto_compressor(self.model, self.args, optimizer_data,
                                     validation_fn_list, test_fn,
                                     save_checkpoint_fn, train_fn)


def main():
    import ea_args
    # Parse arguments
    args = classifier.init_classifier_compression_arg_parser()
    args = ea_args.add_ea_args(args).parse_args()
    app = AutoCompressionSampleApp(args, script_dir=os.path.dirname(__file__))
    return app.train_auto_compressor()


def train_auto_compressor(model, args, optimizer_data, validation_fn_list,
                          test_fn, save_checkpoint_fn, train_fn):
    dataset = args.dataset
    arch = args.arch
    num_ft_epochs = args.ea_ft_epochs
    action_range = args.ea_action_range

    config_verbose(False)

    # Read the experiment configuration
    ea_cfg_fname = args.ea_cfg_file
    if not ea_cfg_fname:
        raise ValueError(
            "You must specify a valid configuration file path using --ea-cfg")

    with open(ea_cfg_fname, 'r') as cfg_file:
        compression_cfg = distiller.utils.yaml_ordered_load(cfg_file)

    services = distiller.utils.MutableNamedTuple({
        'validate_fn_list': validation_fn_list,
        'test_fn': test_fn,
        'save_checkpoint_fn': save_checkpoint_fn,
        'train_fn': train_fn
    })

    app_args = distiller.utils.MutableNamedTuple({
        'dataset': dataset,
        'arch': arch,
        'optimizer_data': optimizer_data,
        'seed': args.seed
    })

    ea_cfg = distiller.utils.MutableNamedTuple({
        'modules_dict':
        compression_cfg["network"],  # dict of modules, indexed by arch name
        'save_chkpts':
        args.ea_save_chkpts,
        'protocol':
        args.ea_protocol,
        'agent_algo':
        args.ea_agent_algo,
        'num_ft_epochs':
        num_ft_epochs,
        'action_range':
        action_range,
        'reward_frequency':
        args.ea_reward_frequency,
        'ft_frequency':
        args.ea_ft_frequency,
        'pruning_pattern':
        args.ea_prune_pattern,
        'pruning_method':
        args.ea_prune_method,
        'group_size':
        args.ea_group_size,
        'n_points_per_fm':
        args.ea_fm_reconstruction_n_pts,
        'ranking_noise':
        args.ea_ranking_noise
    })

    #net_wrapper = NetworkWrapper(model, app_args, services)
    # return sample_networks(net_wrapper, services)

    ea_cfg.target_density = args.ea_target_density
    ea_cfg.reward_fn, ea_cfg.action_constrain_fn = reward_factory(
        args.ea_protocol)

    def create_environment():
        env = Environment(model, app_args, ea_cfg, services)
        return env

    env = create_environment()

    from ea_libs import geatpy_if
    ea = geatpy_if.EALibInterface()
    ea.solve(env, args)
    env.run_test()


def config_verbose(verbose, display_summaries=False):
    if verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
        logging.getLogger().setLevel(logging.WARNING)
    for module in [
            "distiller.apputils.image_classifier", "distiller.thinning",
            "distiller.pruning.ranked_structures_pruner"
    ]:
        logging.getLogger(module).setLevel(loglevel)
        logging.getLogger().setLevel(loglevel)

    # display training progress summaries
    summaries_lvl = logging.INFO if display_summaries else logging.WARNING
    logging.getLogger("examples.auto_compression.amc.summaries").setLevel(
        summaries_lvl)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [
                h for h in msglogger.handlers
                if type(h) != logging.StreamHandler
            ]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' +
                           os.path.realpath(msglogger.log_filename))
