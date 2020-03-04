""" 
    Parse the arguments for the Evolution Algorithm.
"""


def add_ea_args(argparser):
    """ 
    Arguments:
        argparser(argparse.ArgumentParser): Existing parser to which to add the arguments
    """
    group = argparser.add_argument_group(
        'Evolution Algorithm Compress Arguments')
    group.add_argument('--ea-cfg',
                       dest='ea_cfg_file',
                       type=str,
                       action='store',
                       help='EAC configuration file')
    group.add_argument('--ea-protocol',
                       choices=[
                           "mac-constrained", "accuracy-guaranteed",
                           "mac-constrained-experimental", "time first"
                       ],
                       default="mac-constrained",
                       help="Compression-policy search protocol")
    group.add_argument(
        '--ea-ft-epochs',
        type=int,
        default=1,
        help='The number of epochs to fine-tune each discovered network')
    group.add_argument('--ea-save-chkpts',
                       action='store_true',
                       default=False,
                       help='Save the checkpoints of all discovered networks')
    group.add_argument('--ea-action-range',
                       type=float,
                       nargs=2,
                       default=[0.2, 1],
                       help='Density action range (a_min, a_max)')
    group.add_argument('--ea-encoding-type',
                       choices=["RI", "BG", "P"],
                       default="RI",
                       help='Encoding-Type of the Phen')
    group.add_argument('--ea-population-size',
                       type=int,
                       default=50,
                       help="The population size")
    group.add_argument('--ea-max-gen',
                       type=int,
                       default=1000,
                       help="The maximum generation steps")
    group.add_argument('--ea-algorithm',
                       choices=["soea_DE_best_1_L_templet"],
                       default="soea_DE_best_1_L_templet",
                       help="Choose the Evolution algorithm template")
    group.add_argument(
        '--ea-mutOper-F',
        type=float,
        default=0.5,
        help="The Variation scaling factor for differential evolution")
    group.add_argument('--ea-recOper-XOVR',
                       type=float,
                       default=0.5,
                       help="The Cross probability")
    group.add_argument('--ea-drawing-type',
                       action='store_true',
                       default=True,
                       help='Draw the learning process or not')
    group.add_argument(
        '--ea-reward-frequency',
        type=int,
        default=None,
        help='Reward computation frequency(measured in agent steps)')
    group.add_argument('--ea-target-density',
                       type=float,
                       default=0.5,
                       help='Target density of the network we are searching')
    group.add_argument('--ea-agent-algo',
                       choices=["DE", "EGA", "GGAP"],
                       default="DE",
                       help='The agent algorithm to use')
    group.add_argument('--ea-ft-frequency',
                       type=int,
                       default=None,
                       help='How many action-steps between fine-tuning.\n'
                       'By default there is no fine-tuning between steps')
    group.add_argument('--ea-prune-pattern',
                       choices=["filters", "channels", "elements"],
                       default="channels",
                       help='The prunning pattern')
    group.add_argument(
        '--ea-prune-method',
        choices=["l1-rank", "stochastic-l1-rank", "fm-reconstruction"],
        default="l1-rank",
        help='The prunning method')
    group.add_argument('--ea-group-size',
                       type=int,
                       default=1,
                       help='Number of filters/channels to group')
    group.add_argument('--ea-reconstruct-pts',
                       dest='ea_fm_reconstruction_n_pts',
                       type=int,
                       default=10,
                       help='Number of filters/channels to group')
    group.add_argument('--ea-ranking-noise',
                       type=float,
                       default=0.,
                       help='Structure ranking noise')
    return argparser