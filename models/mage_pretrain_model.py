import os
import time
from argparse import Namespace

import torch

from einops import rearrange
import common

from utils.logger_utils import tb_writer, loss_printer
from utils.evaluation import evaluate_mage_pretrain
from utils.train_utils import state_dict_saver, ckpt_saver

from arch.conditioned_mage_arch import DoubleConditionedMAGE
from .modules.masking import Masking
from .basic_model import BasicModel


class MageModel(BasicModel):

    def __init__(self,
                 model: DoubleConditionedMAGE,
                 optimizer,
                 scheduler,
                 criteria,
                 train_data_loader,
                 eval_data_loaders,
                 logger,
                 args: Namespace,
                 writer=None
                 ) -> None:
        super().__init__(total_iterations=args.total_iterations)

        self.logger = logger
        self.args = args
        self.writer = writer

        self.local_rank = args.local_rank

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.train_data_loader = train_data_loader
        self.eval_data_loaders = eval_data_loaders

        self.use_amp = args.get('use_amp', False)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.no_ddp_model = self.model_no_ddp(model)

        # self.ema_model = self.create_ema(model, power=0.75)

    def compile_model(self):
        self.compile(self.model)

    def training_epoch(self, epoch):
        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.model.train()
        nb = len(self.train_data_loader)
        log_vars = {'@loss': None, '@lr': None}
        start_time = time.monotonic()
        for batch_idx, batch in enumerate(self.train_data_loader, start=1):

            total_batches = (epoch - 1) * nb + batch_idx
            # typically we pretrain model with image only dataset, can be same as VQGAN
            x, y = batch

            x = x.to(self.local_rank, non_blocking=True)
            if x.dim() == 5 and x.size(1) == 6:
                x, ref = map(lambda data: rearrange(data, 'b c t h w -> (b t) c h w'),
                             torch.split(x, 3, dim=1))

            assert x.dim() == 4 and x.size(1) == 3, f"Expected get BCHW input shape, but got {x.shape}"

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                loss, imgs, token_all_mask = self.model(x)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # loss.backward()
            # self.optimizer.step()
            self.scheduler.step()

            log_vars['@lr'] = self.scheduler.get_last_lr()[0]
            log_vars['@loss'] = loss

            log_vars = self.reduce_loss_dict(log_vars)

            # self.ema_model.update()

            self.eta_timer.update()
            losses_meter.update(log_vars, x.size(0))

            if batch_idx % self.args.log_steps == 0:
                tb_writer(writer=self.writer, loss_dict=log_vars, nb=total_batches, tag='train')
                s = f"Epoch:{epoch:{' '}{'>'}{2}d}/{self.args.epochs} " \
                    f"iter:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb:.02%}) " \
                    f"est. {self.eta_timer.est(total_batches)} {loss_printer(log_vars, fmt='.04e')}"
                self.logger.info(s)
        self.logger.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{self.args.epochs} finished. Loss: {losses_meter.avg}")

    def evaluating_epoch(self, epoch):
        evaluate_mage_pretrain.evaluation(model=self.model, eval_data_loaders=self.eval_data_loaders,
                                          criteria=self.criteria,
                                          epoch=epoch,
                                          writer=self.writer, args=self.args, logger=self.logger)

    def save_model(self, path, *args):
        state_dict_saver(os.path.join(path, f"{self.no_ddp_model}.pt"), self.no_ddp_model)

    def save_ckpt(self, path, epoch):
        ckpt_saver(os.path.join(path, "latest.pt"),
                   model=self.no_ddp_model,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   epoch=epoch)
