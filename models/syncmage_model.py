import os
from argparse import Namespace

import torch
from einops import rearrange
import common

from utils.logger_utils import tb_writer, loss_printer
from utils.evaluation import evaluate_sync_mage
from utils.train_utils import state_dict_saver, ckpt_saver

from .modules.masking import Masking
from .basic_model import BasicModel


class SyncMageModel(BasicModel):

    def __init__(self,
                 model,
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

        self.no_ddp_model = self.model_no_ddp(model)

        self.mask = Masking(half_precision=True, norm=False).to(self.local_rank)

        self.use_amp = args.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # self.ema_model = self.create_ema(model, power=0.75)

    def compile_model(self):
        self.compile(self.model)

    def training_epoch(self, epoch):
        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.model.train()
        nb = len(self.train_data_loader)
        log_vars = {'@loss': None, '@lr': None}
        for batch_idx, batch in enumerate(self.train_data_loader, start=1):
            total_batches = (epoch - 1) * nb + batch_idx

            x, indiv_mels, mel, y = batch

            x = x.to(self.local_rank, non_blocking=True)

            # REF is the frame from the same video clip with X but not identical
            x, ref = map(lambda data: rearrange(data, 'b c t h w -> (b t) c h w'),
                         torch.split(x, 3, dim=1))

            assert x.dim() == 4 and x.size(1) == 3, f"Expected get BCHW input shape, but got {x.shape}"

            indiv_mels = indiv_mels.to(self.local_rank, non_blocking=True)
            audio_mel = rearrange(indiv_mels, 'b t c h w -> (b t) c h w')

            # unused for now
            # mel = mel.to(self.local_rank, non_blocking=True)

            y = y.to(self.local_rank, non_blocking=True)
            y = rearrange(y, 'b t -> (b t)')

            x_masked = self.mask(x.clone(), mask_face=False)

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                a, v = self.model(x_masked, gt=x, audio=audio_mel)

                loss = self.criteria['sync_loss'](a, v, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # loss.backward()
            #
            # self.optimizer.step()
            self.scheduler.step()

            log_vars['sync_loss'] = loss
            log_vars['@lr'] = self.scheduler.get_last_lr()[0]
            log_vars['@loss'] = loss

            log_vars = self.reduce_loss_dict(log_vars)

            # EMA model update for stable results
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
        return evaluate_sync_mage.evaluation(model=self.model,
                                             eval_data_loaders=self.eval_data_loaders,
                                             epoch=epoch,
                                             criteria=self.criteria,
                                             writer=self.writer,
                                             args=self.args,
                                             logger=self.logger,
                                             mask=self.mask)

    def save_model(self, path, best=False):
        if best:
            state_dict_saver(
                os.path.join(path, f"{self.no_ddp_model}_best.pt"), self.no_ddp_model)
        else:
            state_dict_saver(
                os.path.join(path, f"{self.no_ddp_model}.pt"), self.no_ddp_model)

    def save_ckpt(self, path, epoch):
        ckpt_saver(os.path.join(path, "latest.pt"),
                   model=self.no_ddp_model,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   epoch=epoch)
