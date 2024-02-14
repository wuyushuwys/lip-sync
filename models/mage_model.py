import os
import time
from argparse import Namespace

import torch
from einops import rearrange
import common

from utils.logger_utils import tb_writer, loss_printer
from utils.evaluation import evaluate_mage
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
        super().__init__()

        self.logger = logger
        self.args = args
        self.writer = writer

        self.local_rank = args.local_rank

        self.model: DoubleConditionedMAGE = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.train_data_loader = train_data_loader
        self.eval_data_loaders = eval_data_loaders

        self.mask = Masking(half_precision=True).to(self.local_rank)

        self.use_amp = args.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # self.ema_model = self.create_ema(model, power=0.75)

    def training_epoch(self, epoch):
        time_meter = common.meters.TimeMeter()
        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.model.train()
        nb = len(self.train_data_loader)
        log_vars = {'@loss': None, '@lr': None}

        start_time = time.monotonic()
        for batch_idx, batch in enumerate(self.train_data_loader, start=1):
            self.data_timer.update(time.monotonic() - start_time)

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
            y = rearrange(y, 'b c t h w -> (b t) c h w')

            # mask face
            x_masked = self.mask(x.clone())

            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                loss, imgs, token_all_mask = self.model(x_masked, gt=y, ref=ref, audio=audio_mel)

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

            time_meter.update()
            losses_meter.update(log_vars, x.size(0))

            if batch_idx % self.args.log_steps == 0:
                time_meter.complete_time(nb - batch_idx)
                tb_writer(writer=self.writer, loss_dict=log_vars, nb=total_batches, tag='train')
                s = f"Epoch:{epoch:{' '}{'>'}{2}d}/{self.args.epochs} " \
                    f"iter:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb:.02%}) " \
                    f"est. {time_meter.remain_time} data:{self.data_timer.avg * 1000:.02f}ms {loss_printer(log_vars, fmt='.04e')}"
                self.logger.info(s)

            start_time = time.monotonic()
        self.logger.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{self.args.epochs} finished. SyncLoss: {losses_meter.avg}")

    def evaluating_epoch(self, epoch):
        evaluate_mage.evaluation(model=self.model,
                                 eval_data_loaders=self.eval_data_loaders,
                                 epoch=epoch,
                                 criteria=self.criteria,
                                 writer=self.writer,
                                 args=self.args,
                                 logger=self.logger,
                                 mask=self.mask)

    def save_model(self, path, *args):
        state_dict_saver(os.path.join(path, f"{self.model_no_ddp(self.model)}.pt"), self.model)

    def save_ckpt(self, path, epoch):
        ckpt_saver(os.path.join(path, "latest.pt"),
                   model=self.model,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   epoch=epoch)
