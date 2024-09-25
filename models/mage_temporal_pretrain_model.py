import os
import time

import torch

from einops import rearrange
import common

from utils.logger_utils import tb_writer, loss_printer
from .evaluation import evaluate_mage_temporal_pretrain
from utils.train_utils import state_dict_saver, ckpt_saver

from arch.conditioned_temporal_mage_arch import DoubleTemporalConditionedMAGE
from .basic_model import BasicModel


class MageModel(BasicModel):

    def __init__(self,
                 opt,
                 model: DoubleTemporalConditionedMAGE,
                 optimizer,
                 scheduler,
                 criteria,
                 train_data_loader,
                 eval_data_loaders,
                 writer=None
                 ) -> None:
        super().__init__(opt, total_iterations=opt.total_iterations)

        self.writer = writer

        self.local_rank = opt.local_rank

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.train_data_loader = train_data_loader
        self.eval_data_loaders = eval_data_loaders

        self.use_amp = opt.get('use_amp', False)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.no_ddp_model = self.model_no_ddp(model)

        # self.ema_model = self.create_ema(model, power=0.75)

    def training_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compile_model(self):
        self.training_step = torch.compile(self.training_step)

    def training_epoch(self, epoch):
        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.model.train()
        nb = len(self.train_data_loader)
        log_vars = {}
        start_time = time.monotonic()
        for batch_idx, batch in enumerate(self.train_data_loader, start=1):

            total_batches = (epoch - 1) * nb + batch_idx

            x, _, _, _ = batch
            bsz = x.size(0)
            x = x.to(self.local_rank, non_blocking=True)

            x, ref = map(lambda data: rearrange(data, 'b c t h w -> (b t) c h w'),
                         torch.split(x, 3, dim=1))

            assert x.dim() == 4 and x.size(1) == 3, f"Expected get BCHW input shape, but got {x.shape}"

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                loss, _, _ = self.training_step(x, num_batch=bsz)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            log_vars['@lr'] = self.scheduler.get_last_lr()[0]
            log_vars['@loss'] = loss

            log_vars = self.reduce_loss_dict(log_vars)

            # self.ema_model.update()

            self.eta_timer.update()
            losses_meter.update(log_vars, x.size(0))
            tb_writer(writer=self.writer, loss_dict=log_vars, nb=total_batches, tag='train')
            if batch_idx % self.opt.log_steps == 0:
                s = f"Epoch:{epoch:{' '}{'>'}{2}d}/{self.opt.epochs} " \
                    f"iter:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb:.02%}) " \
                    f"est. {self.eta_timer.est(total_batches)} {loss_printer(log_vars, fmt='.04e')}"
                self.logger.info(s)
        self.logger.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{self.opt.epochs} finished. Loss: {losses_meter.avg}")

    def evaluating_epoch(self, epoch):
        evaluate_mage_temporal_pretrain.evaluation(model=self.model, eval_data_loaders=self.eval_data_loaders,
                                                   criteria=self.criteria,
                                                   epoch=epoch,
                                                   writer=self.writer, args=self.opt, logger=self.logger)

    def save_model(self, path, *opt):
        state_dict_saver(os.path.join(path, f"{self.no_ddp_model}.pt"), self.no_ddp_model)

    def save_ckpt(self, path, epoch):
        ckpt_saver(os.path.join(path, "latest.pt"),
                   model=self.no_ddp_model,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   epoch=epoch)
