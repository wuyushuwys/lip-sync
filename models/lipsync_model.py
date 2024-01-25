import os
from argparse import Namespace

import common

from utils.logger_utils import tb_writer, loss_printer
from utils.evaluation import evaluate_lip
from utils.train_utils import state_dict_saver, ckpt_saver

from .modules.masking import Masking
from .basic_model import BasicModel


class LipSyncModel(BasicModel):

    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 criterion,
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

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        self.eval_data_loaders = eval_data_loaders

        self.mask = Masking(half_precision=True).to(self.local_rank)

        self.ema_model = self.create_ema(model)

    def training_epoch(self, epoch):
        time_meter = common.meters.TimeMeter()
        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.model.train()
        nb = len(self.train_data_loader)
        log_vars = {'@loss': None, '@lr': None}
        for batch_idx, batch in enumerate(self.train_data_loader, start=1):
            total_batches = (epoch - 1) * nb + batch_idx

            x, indiv_mels, mel, y = batch

            x = x.to(self.local_rank, non_blocking=True)
            indiv_mels = indiv_mels.to(self.local_rank, non_blocking=True)
            mel = mel.to(self.local_rank, non_blocking=True)
            y = y.to(self.local_rank, non_blocking=True)

            # mask face
            x = self.mask(x)

            self.optimizer.zero_grad()

            pred_y = self.model(indiv_mels, x)

            sync_weight = self.criterion['sync_loss'].loss_weight
            sync_loss = self.criterion['sync_loss'](mel, pred_y) if sync_weight != 0 else 0

            recon_loss = self.criterion['recon_loss'](pred_y, y) if 'recon_loss' in self.criterion.keys() else 0

            perceptual_loss = self.criterion['perceptual_loss'](pred_y,
                                                                y) if 'perceptual_loss' in self.criterion.keys() else 0

            loss = sync_loss * sync_weight + (recon_loss + perceptual_loss) * (1 - sync_weight)

            loss.backward()
            log_vars['sync_loss'] = sync_loss
            log_vars['recon_loss'] = recon_loss
            log_vars['perceptual_loss'] = perceptual_loss
            log_vars['@lr'] = self.scheduler.get_last_lr()[0]
            log_vars['@loss'] = loss

            self.optimizer.step()
            self.scheduler.step()

            self.ema_model.update()

            time_meter.update()
            losses_meter.update(log_vars, x.size(0))

            if batch_idx % self.args.log_steps == 0:
                time_meter.complete_time(nb - batch_idx)
                tb_writer(writer=self.writer, loss_dict=log_vars, nb=total_batches, tag='train')
                s = f"Epoch:{epoch:{' '}{'>'}{2}d}/{self.args.epochs} " \
                    f"iter:{batch_idx:{' '}{'>'}{len(str(nb))}d}/{nb:d}({batch_idx / nb:.02%}) " \
                    f"est. {time_meter.remain_time} {loss_printer(log_vars, fmt='.04e')}"
                self.logger.info(s)
        self.logger.info(f"Epoch{epoch:{' '}{'>'}{2}d}/{self.args.epochs} finished. SyncLoss: {losses_meter.avg}")

    def evaluating_epoch(self, epoch):
        sync_loss = evaluate_lip.evaluation(model=self.ema_model,
                                            eval_data_loaders=self.eval_data_loaders,
                                            epoch=epoch,
                                            criterions=self.criterion,
                                            writer=self.writer,
                                            args=self.args,
                                            logger=self.logger,
                                            mask=self.mask)
        if sync_loss < 0.75:
            self.criterion['sync_loss'].loss_weight = 0.03

    def save_model(self, path, *args):
        state_dict_saver(os.path.join(path, f"{self.model_no_ddp(self.model)}.pt"), self.ema_model.ema_model)

    def save_ckpt(self, path, epoch):
        ckpt_saver(os.path.join(path, "latest.pt"),
                   model=self.ema_model.ema_model,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   epoch=epoch)
