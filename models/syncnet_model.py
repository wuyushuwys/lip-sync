import os

import torch
import common

from utils.logger_utils import tb_writer, loss_printer
from .evaluation import evaluate_sync
from utils.train_utils import state_dict_saver, ckpt_saver

from .modules.masking import Masking
from .basic_model import BasicModel


class SyncNetModel(BasicModel):

    def __init__(self,
                 opt,
                 model,
                 optimizer,
                 scheduler,
                 criteria,
                 train_data_loader,
                 eval_data_loaders,
                 writer=None
                 ) -> None:
        super().__init__(opt=opt, total_iterations=opt.total_iterations)

        self.writer = writer

        self.local_rank = opt.local_rank

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.train_data_loader = train_data_loader
        self.eval_data_loaders = eval_data_loaders

        self.no_ddp_model = self.model_no_ddp(model)

        mask_kwargs = opt.get("mask", dict(half_precision=True, norm=False))
        self.mask = Masking(**mask_kwargs).to(self.local_rank)

        self.cur_loss = None
        self.best_loss = float('inf')

        self.use_amp = opt.get('use_amp', False)
        self.bottom_half = opt.video_spec.get('bottom_half', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def compile_model(self):
        self.compile(self.model)

    def training_epoch(self, epoch):
        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.model.train()
        nb = len(self.train_data_loader)
        log_vars = {}
        for batch_idx, batch in enumerate(self.train_data_loader, start=1):
            total_batches = (epoch - 1) * nb + batch_idx

            x, mel, y = batch
            x = x.to(self.local_rank, non_blocking=True)
            mel = mel.to(self.local_rank, non_blocking=True)
            y = y.to(self.local_rank, non_blocking=True)

            with torch.no_grad():
                if not self.bottom_half:
                    x = self.mask(x.clone(), mask_face=False, lip_only=True)

            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                a, v = self.model(mel, x)

                loss = self.criteria['sync_loss'](a, v, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)

            self.scaler.update()
            self.scheduler.step()

            log_vars['sync_loss'] = loss
            log_vars['@lr'] = self.scheduler.get_last_lr()[0]
            log_vars['@loss'] = loss

            log_vars = self.reduce_loss_dict(log_vars)

            # EMA model update for stable results
            # self.ema_model.update()

            self.eta_timer.update()
            losses_meter.update(log_vars, x.size(0))

            if batch_idx % self.opt.log_steps == 0:
                tb_writer(writer=self.writer, loss_dict=log_vars, nb=total_batches, tag='train')
                s = f"Epoch:{epoch:{' '}{'>'}{2}d}/{self.opt.epochs} " \
                    f"iter:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb:.02%}) " \
                    f"est. {self.eta_timer.est(total_batches)} {loss_printer(log_vars, fmt='.04e')}"
                self.logger.info(s)
        self.logger.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{self.opt.epochs} finished. Loss: {losses_meter.avg}")

    def evaluating_epoch(self, epoch):
        self.cur_loss = evaluate_sync.evaluation(model=self.model,
                                                 eval_data_loaders=self.eval_data_loaders,
                                                 epoch=epoch,
                                                 criteria=self.criteria,
                                                 writer=self.writer,
                                                 args=self.opt,
                                                 logger=self.logger, mask=self.mask)

    def save_model(self, path, best=False):

        if self.cur_loss < self.best_loss:
            self.best_loss = self.cur_loss
            self.logger.info('Save best model weights')
            state_dict_saver(
                os.path.join(path, f"{self.no_ddp_model}_best.pt"), self.no_ddp_model)
        state_dict_saver(
            os.path.join(path, f"{self.no_ddp_model}.pt"), self.no_ddp_model)

    def save_ckpt(self, path, epoch):
        ckpt_saver(os.path.join(path, "latest.pt"),
                   model=self.no_ddp_model,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   epoch=epoch)
