import os

import torch
from einops import rearrange
import common

from utils.logger_utils import tb_writer, loss_printer
from utils.train_utils import state_dict_saver, ckpt_saver

from arch.ref_control_net_arch import RefControlNet
from arch.discriminator_arch import UNetDiscriminatorSN

from .evaluation import evaluate_ref_control
from arch.modules.masking import Masking
from .basic_model import BasicModel


class RefControlGANModel(BasicModel):

    def __init__(self,
                 opt,
                 g_model: RefControlNet,
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
        super().__init__(opt=opt, total_iterations=opt.total_iterations)

        self.writer = writer

        self.local_rank = opt.local_rank

        self.g_model: RefControlNet = g_model
        self.g_optimizer = g_optimizer
        self.g_scheduler = g_scheduler

        self.d_model: UNetDiscriminatorSN = d_model
        self.d_optimizer = d_optimizer
        self.d_scheduler = d_scheduler

        self.criteria = criteria
        self.train_data_loader = train_data_loader
        self.eval_data_loaders = eval_data_loaders

        mask_kwargs = opt.get("mask", dict(half_precision=True, norm=False))
        self.mask = Masking(**mask_kwargs).to(self.local_rank)

        self.use_amp = opt.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.curr_iterations = 0
        if opt.get('use_gan', False):
            self.gan_starts = int(opt.total_iterations * opt.gan_starts) if 0 <= opt.gan_starts <= 1 else opt.gan_starts
        else:
            self.gan_starts = int(opt.total_iterations)
        self.clip_grad = opt.get('clip_grad', False)

        self.no_ddp_g_model = self.model_no_ddp(g_model)
        self.no_ddp_d_model = self.model_no_ddp(d_model)

    def init_trainer(self):
        pass

    def compile_model(self):
        self.g_model, self.d_model = self.compile(self.g_model, self.d_model)

    def training_epoch(self, epoch):

        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.g_model.train()
        nb = len(self.train_data_loader)
        log_vars = {}

        for batch_idx, batch in enumerate(self.train_data_loader, start=1):

            total_batches = (epoch - 1) * nb + batch_idx

            self.curr_iterations = total_batches

            x, indiv_mels, mel, y = batch

            x = x.to(self.local_rank, non_blocking=True)

            # REF is the frame from the same video clip with X but not identical
            x, ref = map(lambda data: rearrange(data, 'b c t h w -> (b t) c h w'),
                         torch.split(x, 3, dim=1))

            assert x.dim() == 4 and x.size(1) == 3, f"Expected get BCHW input shape, but got {x.shape}"

            # indiv_mels = indiv_mels.to(self.local_rank, non_blocking=True)
            # audio_mel = rearrange(indiv_mels, 'b t c h w -> (b t) c h w')

            # unused for now
            # mel = mel.to(self.local_rank, non_blocking=True)

            y = y.to(self.local_rank, non_blocking=True)
            y = rearrange(y, 'b c t h w -> (b t) c h w')

            # mask face
            # with torch.no_grad():
            #     x_masked = self.mask(x.clone())

            ############################################
            # optimize generator
            ############################################

            for p in self.d_model.parameters():
                p.requires_grad = False

            self.g_optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                g = self.g_model(x, ref)

                g_loss = 0
                if 'recon_loss' in self.criteria.keys():
                    pixel_loss = self.criteria['recon_loss'](g, y)
                    g_loss += pixel_loss
                else:
                    pixel_loss = 0
                if 'perceptual_loss' in self.criteria.keys():
                    perceptual_loss = self.criteria['perceptual_loss'](g, y)
                    g_loss += perceptual_loss
                else:
                    perceptual_loss = 0

                if self.curr_iterations > self.gan_starts:
                    fake_g_pred = self.d_model(g)

                    adversarial_loss = self.criteria['adversarial'](fake_g_pred, True, is_disc=False)
                    g_loss += adversarial_loss

                self.scaler.scale(g_loss).backward()

                if self.clip_grad:
                    self.scaler.unscale_(self.g_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.no_ddp_g_model.parameters(), self.clip_grad, foreach=True)

                self.scaler.step(self.g_optimizer)
                self.scaler.update()
            self.g_scheduler.step()

            if self.curr_iterations > self.gan_starts:
                ############################################
                # optimize discriminator
                ############################################

                for p in self.d_model.parameters():
                    p.requires_grad = True

                self.d_optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):

                    real_d_pred = self.d_model(y.contiguous().detach())
                    l_d_real = self.criteria['adversarial'](real_d_pred, True, is_disc=True)
                    self.scaler.scale(l_d_real).backward()

                    fake_d_pred = self.d_model(g.contiguous().detach())
                    l_d_fake = self.criteria['adversarial'](fake_d_pred, False, is_disc=True)
                    self.scaler.scale(l_d_fake).backward()

                    if self.clip_grad:
                        self.scaler.unscale_(self.d_optimizer)
                        torch.nn.utils.clip_grad_norm_(self.no_ddp_d_model.parameters(), self.clip_grad, foreach=True)
                    self.scaler.step(self.d_optimizer)
                    self.scaler.update()
                # self.d_optimizer.step()
                self.d_scheduler.step()

            log_vars['@lr'] = self.g_scheduler.get_last_lr()[0]
            log_vars['@g_loss'] = g_loss
            log_vars['recon_loss'] = pixel_loss
            log_vars['perceptual_loss'] = perceptual_loss

            if self.curr_iterations > self.gan_starts:
                log_vars['adversarial_loss'] = adversarial_loss
                log_vars['d_real'] = l_d_real
                log_vars['d_fake'] = l_d_fake
                log_vars['@d_loss'] = l_d_real + l_d_real

            log_vars = self.reduce_loss_dict(log_vars)

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
        evaluate_ref_control.evaluation(model=self.g_model,
                                        eval_data_loaders=self.eval_data_loaders,
                                        epoch=epoch,
                                        criteria=self.criteria,
                                        writer=self.writer,
                                        args=self.opt,
                                        logger=self.logger,
                                        mask=self.mask)

    def save_model(self, path, *args):
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
