__all__ = ['arguments_parser']


def arguments_parser(parser):
    parser.add_argument('--config_file', default=None, type=str, required=True)

    # Dataset
    parser.add_argument('--dataset', default=None, type=str, required=True,
                        help='Dataset name.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training and evaluation.')
    # parser.add_argument('--eval_batch_size', default=256, type=int,
    #                     help='Batch size for evaluation.')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers for data loading.')

    parser.add_argument('--job_dir', default=None, type=str,
                        help='Directory to write checkpoints and export models.')
    parser.add_argument('--ckpt', default=None, type=str,
                        help='Dir path to load gan checkpoint.')
    parser.add_argument('--g_weight', default=None, type=str,
                        help='path to generator weight')
    parser.add_argument('--profile', action='store_true',
                        help='profile model')
    # parser.add_argument('--prefetcher', action='store_true',
    #                     help='using data prefetcher')

    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    # evaluation
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help='Running evaluation only.')
    parser.add_argument('--eval_model', default=None, type=str,
                        help='Path to evaluation model.')
    parser.add_argument('--eval_datasets', default=None, type=str, nargs='+',
                        help='Dataset names for evaluation.')
    parser.add_argument('--lpips_arch', default='alex', type=str, choices=['vgg', 'alex'],
                        help='lpips evaluation architecture.')

    # Experiment arguments
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs to train gan.')
    parser.add_argument('--log_steps', default=0, type=int,
                        help='Number of steps for training logging.')
    parser.add_argument('--opt_level', default='O1', type=str,
                        help='Recognized opt_levels are O0, O1, O2, and O3')

    parser.add_argument('--resume', default=False, action='store_true',
                        help='resume GAN training from gan_ckpt_path')
    parser.add_argument('--warmup_lr', default=False, action='store_true',
                        help='Using warmup lr')

    # gradient clipper
    parser.add_argument('--clip_grad', default=False, action='store_true',
                        help='Using gradient clipper to avoid gradient explosive or diminish')
    parser.add_argument('--gradient_range', default=5, type=float,
                        help='Gradient clip range')
    parser.add_argument('--clip_epoch', default=0.1, type=float,
                        help='% of epochs to perform gradient clipper')

    # Verbose
    parser.add_argument('-v', '--verbose', action='count', default=1,
                        help='Increasing output verbosity.', )
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug mode. Enable torch anomaly check')

    # manual seed
    parser.add_argument('--manual_seed', default=False, action='store_true',
                        help='use manual seeds')
