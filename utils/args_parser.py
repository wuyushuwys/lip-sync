__all__ = ['arguments_parser']


def arguments_parser(parser):
    parser.add_argument('--config_file', default=None, type=str, required=True,
                        help='path to config file')

    parser.add_argument('--model', default=None, type=str, required=True,
                        choices=['syncnet', 'lipsync', 'vqgan', 'mage'],
                        help='model to train')

    # Dataset
    parser.add_argument('--dataset', default=None, type=str, required=True, nargs='+',
                        help='Dataset name.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training and evaluation.')
    # parser.add_argument('--train_batch_size', default=256, type=int,
    #                     help='Batch size for training.')
    # parser.add_argument('--eval_batch_size', default=256, type=int,
    #                     help='Batch size for evaluation.')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers for data loading.')

    parser.add_argument('--job_dir', default=None, type=str,
                        help='Directory to write ckpt and export models.')
    parser.add_argument('--ckpt', default=None, type=str,
                        help='Dir path to load ckpt.')
    parser.add_argument('--weight', default=None, type=str,
                        help='path to load model weight')
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

    # Experiment arguments
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs to train gan.')
    parser.add_argument('--log_steps', default=0, type=int,
                        help='Number of steps for training logging.')
    # parser.add_argument('--log_scale', default=10, type=int,
    #                     help='Scale for logging per epoch.e.g., log_steps = ')

    parser.add_argument('--warmup_lr', default=False, action='store_true',
                        help='Using warmup lr')

    parser.add_argument('--scale_lr', default=False, action='store_true',
                        help='Using scaled lr in multi-gpu')

    # gradient clipper
    # parser.add_argument('--clip_grad', default=False, action='store_true',
    #                     help='Using gradient clipper to avoid gradient explosive or diminish')
    # parser.add_argument('--gradient_range', default=5, type=float,
    #                     help='Gradient clip range')
    # parser.add_argument('--clip_epoch', default=0.1, type=float,
    #                     help='% of epochs to perform gradient clipper')

    # Verbose
    # parser.add_argument('-v', '--verbose', action='count', default=1,
    #                     help='Increasing output verbosity.', )
    # parser.add_argument('--debug', default=False, action='store_true',
    #                     help='debug mode. Enable torch anomaly check')

    # manual seed
    parser.add_argument('--manual_seed', default=False, action='store_true',
                        help='use manual seeds')
