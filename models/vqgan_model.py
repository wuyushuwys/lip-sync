import os
import wandb

import torch

import common

from utils import master_only, state_dict_saver, ckpt_saver, tb_writer, loss_printer
from .evaluation import evaluate_vq

from .basic_model import BasicModel


class VQGANModel(BasicModel):

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
        super().__init__(opt=opt, total_iterations=opt.total_iterations)

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

        # self.ema_g_model = self.create_ema(self.g_model, power=0.75)
        # self.ema_d_model = self.create_ema(self.d_model, power=0.75)

        self.curr_iterations = 0
        self.gan_starts = int(opt.total_iterations * opt.gan_starts)
        self.codebook_weight = opt.losses.codebook_loss.loss_weight
        self.semantic_weight = opt.losses.semantic_loss.loss_weight if 'semantic_loss' in opt.losses.keys() else None
        self.logger.info(f"Total iterations {opt.total_iterations}, GAN starts at {self.gan_starts}")

        self.clip_grad = opt.get('clip_grad', False)

        self.use_amp = opt.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def compile_model(self):
        self.g_model, self.d_model = self.compile(self.g_model, self.d_model)

    @staticmethod
    def calculate_adaptive_weight(recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        return d_weight

    def training_epoch(self, epoch):

        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.g_model.train()
        self.d_model.train()
        nb = len(self.train_data_loader)
        log_vars = {}

        for batch_idx, batch in enumerate(self.train_data_loader, start=1):

            total_batches = (epoch - 1) * nb + batch_idx
            self.curr_iterations = total_batches

            x, y = batch

            x = x.to(self.local_rank, non_blocking=True)
            y = y.to(self.local_rank, non_blocking=True)

            ############################################
            # optimize generator
            ############################################

            for p in self.d_model.parameters():
                p.requires_grad = False

            self.g_optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                pred_y, vq_info = self.g_model(x)

                if 'recon_loss' in self.criteria.keys():
                    recon_loss = self.criteria['recon_loss'](pred_y, y)
                else:
                    recon_loss = 0

                if 'perceptual_loss' in self.criteria.keys():
                    perceptual_loss = self.criteria['perceptual_loss'](pred_y, y, normalize=False)
                else:
                    perceptual_loss = 0

                g_loss = recon_loss + perceptual_loss + vq_info.codebook_loss * self.codebook_weight

                # if self.model_no_ddp(self.g_model).use_semantic_loss:
                #     g_loss += vq_info.semantic_loss * self.semantic_weight

                if self.curr_iterations > self.gan_starts:
                    fake_g_pred = self.d_model(pred_y)

                    adversarial_loss = self.criteria['adversarial'](fake_g_pred, True, is_disc=False)
                    # adv_weight = self.calculate_adaptive_weight(recon_loss+perceptual_loss,
                    #                                             adversarial_loss,
                    #                                             last_layer=self.g_model.module.generator.blocks[-1].weight,
                    #                                             disc_weight_max=1.0)
                    # g_loss += adversarial_loss * adv_weight
                    g_loss += adversarial_loss
                self.scaler.scale(g_loss).backward()

                # g_loss.backward()

                if self.clip_grad:
                    self.scaler.unscale_(self.g_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.no_ddp_g_model.parameters(), self.clip_grad, foreach=True)

                self.scaler.step(self.g_optimizer)
                self.scaler.update()
                # self.g_optimizer.step()
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
                    # l_d_real.backward()
                    self.scaler.scale(l_d_real).backward()

                    fake_d_pred = self.d_model(pred_y.contiguous().detach())
                    l_d_fake = self.criteria['adversarial'](fake_d_pred, False, is_disc=True)
                    # l_d_fake.backward()
                    self.scaler.scale(l_d_fake).backward()

                    if self.clip_grad:
                        self.scaler.unscale_(self.d_optimizer)
                        torch.nn.utils.clip_grad_norm_(self.no_ddp_d_model.parameters(), self.clip_grad, foreach=True)
                    self.scaler.step(self.d_optimizer)
                    self.scaler.update()
                # self.d_optimizer.step()
                self.d_scheduler.step()

            log_vars['recon_loss'] = recon_loss
            log_vars['perceptual_loss'] = perceptual_loss
            log_vars['@lr'] = self.g_scheduler.get_last_lr()[0]
            log_vars['codebook_loss'] = vq_info.codebook_loss
            log_vars['@g_loss'] = g_loss
            # if self.model_no_ddp(self.g_model).use_semantic_loss:
            #     log_vars['semantic_loss'] = vq_info.semantic_loss

            if self.curr_iterations > self.gan_starts:
                log_vars['adversarial_loss'] = adversarial_loss
                log_vars['d_real'] = l_d_real
                log_vars['real_d_pred'] = real_d_pred.detach().mean()
                log_vars['d_fake'] = l_d_fake
                log_vars['fake_d_pred'] = fake_d_pred.detach().mean()
                log_vars['@d_loss'] = l_d_real + l_d_real
                # self.ema_d_model.update()

            # self.ema_g_model.update()

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
        self.log_codebook()
        evaluate_vq.evaluation(model=self.g_model,
                               eval_data_loaders=self.eval_data_loaders,
                               epoch=epoch,
                               criteria=self.criteria,
                               writer=self.writer,
                               args=self.opt,
                               logger=self.logger)

    @master_only
    def log_codebook(self, up_factor=2):
        codebook, num_code = self.no_ddp_g_model.vis_codebook(up_factor=up_factor)
        if wandb.run is not None:
            image = wandb.Image(codebook, caption=f"num_code-{num_code}", file_type='jpg')
            wandb.log({"Codebook": image})

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
