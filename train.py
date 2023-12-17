import warnings

import argparse
import importlib
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

import wandb

import config

from utils import *
from utils.logging_tool import LoggingTool
from models import Model

save_image = master_only(save_image)
print = master_only(print)


def train(**kwargs):
    pass
    # g_loss_meter = common.meters.AverageMeter()
    # time_meter = common.meters.TimeMeter()
    # losses_meter = common.meters.LossesMeter(fmt='.04e')
    #
    # g_model.train()
    # nb = len(train_data_loader)
    # losses = {}
    # batch_idx = 0
    # train_data_loader.reset()
    # batch = train_data_loader.next()
    # while batch is not None:
    #     batch_idx += 1
    #
    #     total_batches = (epoch - 1) * nb + batch_idx
    #
    #     batch = train_data_loader.next()
    #
    #     time_meter.update()
    #     losses_meter.update(losses)
    #
    #     if batch_idx % params.log_steps == 0 and params.rank == 0:
    #         pass
    # time_meter.complete_time(nb - batch_idx)
    # tb_writer(writer=writer, loss_dict=losses, nb=total_batches, tag='train')
    # writer.add_scalar('train/loss', g_loss_meter.val, total_batches)
    # time_meter.complete_time(nb - batch_idx)
    # s = f"## Epoch:{epoch:{' '}{'>'}{2}d}/{params.epochs}\t" \
    #     f"Iters:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb * 100:.2f}%)\t" \
    #     f"Epoch-est. {time_meter.remain_time}\t" \
    #     f"Loss: {g_loss_meter.val:.04f}\t" \
    #     f"{loss_printer(losses, fmt='.04e')}"
    # print(s)
    # logging.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{params.epochs} finished."
    #              f"\tG_loss: {g_loss_meter.avg:.6f}"
    #              f"\t{losses_meter.print_avg()}")
    # save_image(recon.clamp(0, 1), os.path.join(params.job_dir, 'results', f'epoch_{epoch:02d}_output.bmp'))
    # save_image(real_img, os.path.join(params.job_dir, 'results', f'epoch_{epoch:02d}_target.bmp'))


def main(args):
    # Enable cudnn Optimization for static network structure
    torch.backends.cudnn.benchmark = True

    device = args.local_rank

    # Create job and tb_writer
    writer = SummaryWriter(args.job_dir) if args.rank == 0 else None

    # Load train datasetcd
    train_data_loader, train_sampler, eval_data_loaders, eval_sampler = create_dataloader(args)

    # Create generator
    # model = Model(params)

    logging.info(f"\n{g_model}", is_print=False)

    # profile_model(params)

    # Loss function
    criterion = create_criterions(args)

    # create optimizers and schedulers
    # [], [] = create_optim_scheduler(**kwargs)

    # Load ckpt
    if args.resume and args.ckpt:
        pass
    else:
        pass

    # Load state_dict

    pass

    # allocate model to gpu
    if args.distributed:
        pass
    else:
        pass

    logging.info(attr_extractor(args))

    # Eval model

    # Train
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train()
        # evaluation_generator(g_model, eval_data_loaders, epoch, writer, device=device, params=params)
        # save model weight
        # state_dict_saver(os.path.join(params.job_dir, 'weights', 'g_model.pt'), g_model)
        # ckpt_saver(os.path.join(params.job_dir, "ckpt", f"g_latest.pth"),
        #            g_model=g_model,
        #            g_optimizer=g_optimizer, g_scheduler=g_scheduler,
        #            entropy_optimizer=entropy_optimizer, entropy_scheduler=entropy_scheduler,
        #            epoch=epoch)
    # if params.rank == 0:
    #     writer.close()

    logging.info(f"Finish Training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arguments_parser(parser)

    # Parse arguments
    params = parser.parse_args()
    logging = LoggingTool(file_path=params.job_dir, verbose=params.verbose)
    params.logger = logging
    init_process(params)

    # parsing args
    # params = parser.parse_args(namespace=args)
    config.update_params(params)

    main(params)
