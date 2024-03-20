import os

import torch
from einops import rearrange

import common

from utils.logger_utils import tb_writer, loss_printer
from .evaluation import evaluate_lip
from utils.train_utils import state_dict_saver, ckpt_saver

from .modules.masking import Masking
from .basic_model import BasicModel

face_rearrange = lambda x: rearrange(x, 'b c t h w -> (b t) c h w')


class LipSyncGAN(BasicModel):

    def __init__(self,
                 opt,
                 g_model,
                 g_optimizer,
                 g_scheduler,
                 d_model,
                 d_optimizer,
                 d_scheduler,
                 criteria,
                 train_data_loader,
                 eval_data_loaders,
                 writer=None
                 ) -> None:
        super().__init__(opt, total_iterations=opt.total_iterations)

        self.writer = writer

        self.local_rank = opt.local_rank

        self.g_model = g_model
        self.g_optimizer = g_optimizer
        self.g_scheduler = g_scheduler

        self.d_model = d_model
        self.d_optimizer = d_optimizer
        self.d_scheduler = d_scheduler

        self.criteria = criteria
        self.train_data_loader = train_data_loader
        self.eval_data_loaders = eval_data_loaders

        self.no_ddp_g_model = self.model_no_ddp(g_model)
        self.no_ddp_d_model = self.model_no_ddp(d_model)

        mask_kwargs = opt.get("mask", dict(half_precision=True))
        self.mask = Masking(**mask_kwargs).to(self.local_rank)
        self.use_mask = opt.get('use_mask', False)
        # self.ema_g_model = self.create_ema(self.g_model, power=0.75)
        # self.ema_d_model = self.create_ema(self.d_model, power=0.75)

    def compile_model(self):
        self.compile(self.g_model, self.d_model)

    def training_epoch(self, epoch):

        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.g_model.train()
        self.d_model.train()
        nb = len(self.train_data_loader)
        log_vars = {}

        for batch_idx, batch in enumerate(self.train_data_loader, start=1):

            total_batches = (epoch - 1) * nb + batch_idx

            x, indiv_mels, mel, y = batch

            x = x.to(self.local_rank, non_blocking=True)
            indiv_mels = indiv_mels.to(self.local_rank, non_blocking=True)
            mel = mel.to(self.local_rank, non_blocking=True)
            y = y.to(self.local_rank, non_blocking=True)

            # mask face
            with torch.no_grad():
                x = self.mask(x)

            ############################################
            # optimize generator
            ############################################

            for p in self.d_model.parameters():
                p.requires_grad = False

            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()

            pred_y = self.g_model(indiv_mels, x)

            sync_weight = self.criteria['sync_loss'].loss_weight
            sync_loss = self.criteria['sync_loss'](mel, pred_y,
                                                   mask=self.mask if self.use_mask else None) if sync_weight != 0 else 0

            if 'recon_loss' in self.criteria.keys():
                recon_loss = self.criteria['recon_loss'](pred_y, y)
            else:
                recon_loss = 0

            if 'perceptual_loss' in self.criteria.keys():
                perceptual_loss = self.criteria['perceptual_loss'](pred_y, y)
            else:
                perceptual_loss = 0

            fake_g_pred = self.d_model(face_rearrange(pred_y))

            adversarial_loss = self.criteria['adversarial'](fake_g_pred, True, is_disc=False)

            g_loss = sync_loss * sync_weight + (recon_loss + perceptual_loss + adversarial_loss) * (1 - sync_weight)

            g_loss.backward()
            self.g_optimizer.step()
            self.g_scheduler.step()

            ############################################
            # optimize discriminator
            ############################################

            for p in self.d_model.parameters():
                p.requires_grad = True

            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()

            real_d_pred = self.d_model(face_rearrange(y).contiguous().detach())
            l_d_real = self.criteria['adversarial'](real_d_pred, True, is_disc=True)
            l_d_real.backward()

            fake_d_pred = self.d_model(face_rearrange(pred_y.detach()))
            l_d_fake = self.criteria['adversarial'](fake_d_pred, False, is_disc=True)
            l_d_fake.backward()

            self.d_optimizer.step()
            self.d_scheduler.step()

            # self.ema_g_model.update()
            # self.ema_d_model.update()

            log_vars['sync_loss'] = sync_loss
            log_vars['recon_loss'] = recon_loss
            log_vars['perceptual_loss'] = perceptual_loss
            log_vars['adversarial_loss'] = adversarial_loss
            log_vars['@lr'] = self.g_scheduler.get_last_lr()[0]
            log_vars['@g_loss'] = g_loss
            log_vars['d_real'] = l_d_real
            log_vars['d_fake'] = l_d_fake
            log_vars['@d_loss'] = l_d_real + l_d_real

            log_vars = self.reduce_loss_dict(log_vars)

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
        sync_loss = evaluate_lip.evaluation(model=self.g_model,
                                            eval_data_loaders=self.eval_data_loaders,
                                            epoch=epoch,
                                            criteria=self.criteria,
                                            writer=self.writer,
                                            args=self.opt,
                                            logger=self.logger,
                                            mask=self.mask)
        if sync_loss < 0.75:
            self.criteria['sync_loss'].loss_weight = 0.03

    def save_model(self, path, *opt):

        state_dict_saver(os.path.join(path, f"{self.no_ddp_g_model}.pt"), self.no_ddp_g_model)
        state_dict_saver(os.path.join(path, f"{self.no_ddp_d_model}.pt"), self.no_ddp_d_model)

    def save_ckpt(self, path, epoch):
        ckpt_saver(os.path.join(path, "latest.pt"),
                   g_model=self.no_ddp_g_model,
                   g_optimizer=self.g_optimizer,
                   g_scheduler=self.g_scheduler,
                   d_model=self.no_ddp_d_model,
                   d_optimizer=self.d_optimizer,
                   d_scheduler=self.d_scheduler,
                   epoch=epoch)
