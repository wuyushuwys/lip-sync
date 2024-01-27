import os
from argparse import Namespace
import wandb

import torch

import common

from utils import master_only, state_dict_saver, ckpt_saver, tb_writer, loss_printer
from utils.evaluation import evaluate_vq

from .basic_model import BasicModel


class VQGANModel(BasicModel):

    def __init__(self,
                 g_model,
                 g_optimizer,
                 g_scheduler,
                 d_model,
                 d_optimizer,
                 d_scheduler,
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

        self.g_model = g_model
        self.g_optimizer = g_optimizer
        self.g_scheduler = g_scheduler

        self.d_model = d_model
        self.d_optimizer = d_optimizer
        self.d_scheduler = d_scheduler

        self.criterion = criterion
        self.train_data_loader = train_data_loader
        self.eval_data_loaders = eval_data_loaders

        self.ema_g_model = self.create_ema(self.g_model)
        self.ema_d_model = self.create_ema(self.d_model)

        self.curr_iterations = 0
        self.gan_starts = int(args.total_iterations * args.gan_starts)
        self.codebook_weight = args.losses.codebook_loss.loss_weight
        self.logger.info(f"Total iterations {args.total_iterations}, GAN starts at {self.gan_starts}")

    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        return d_weight

    def training_epoch(self, epoch):
        time_meter = common.meters.TimeMeter()
        losses_meter = common.meters.LossesMeter(fmt='.04e')
        self.g_model.train()
        self.d_model.train()
        nb = len(self.train_data_loader)
        log_vars = {"@g_loss": None, '@lr': None}
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

            pred_y, codebook_loss, quant_stats = self.g_model(x)

            if 'recon_loss' in self.criterion.keys():
                recon_loss = self.criterion['recon_loss'](pred_y, y)
            else:
                recon_loss = 0

            if 'perceptual_loss' in self.criterion.keys():
                perceptual_loss = self.criterion['perceptual_loss'](pred_y, y, normalize=False)
            else:
                perceptual_loss = 0

            g_loss = recon_loss + perceptual_loss + codebook_loss * self.codebook_weight

            if self.curr_iterations > self.gan_starts:
                fake_g_pred = self.d_model(pred_y)

                adversarial_loss = self.criterion['adversarial'](fake_g_pred, True, is_disc=False)
                adv_weight = self.calculate_adaptive_weight(recon_loss+perceptual_loss,
                                                            adversarial_loss,
                                                            last_layer=self.g_model.module.generator.blocks[-1].weight,
                                                            disc_weight_max=1.0)
                g_loss += adversarial_loss * adv_weight
                # g_loss += adversarial_loss

            g_loss.backward()
            self.g_optimizer.step()
            self.g_scheduler.step()

            if self.curr_iterations > self.gan_starts:
                ############################################
                # optimize discriminator
                ############################################

                for p in self.d_model.parameters():
                    p.requires_grad = True

                self.d_optimizer.zero_grad()

                real_d_pred = self.d_model(y)
                l_d_real = self.criterion['adversarial'](real_d_pred, True, is_disc=True)
                l_d_real.backward()

                fake_d_pred = self.d_model(pred_y.detach().clone())
                l_d_fake = self.criterion['adversarial'](fake_d_pred, False, is_disc=True)
                l_d_fake.backward()

                self.d_optimizer.step()
                self.d_scheduler.step()

            log_vars['recon_loss'] = recon_loss
            log_vars['perceptual_loss'] = perceptual_loss
            log_vars['@lr'] = self.g_scheduler.get_last_lr()[0]
            log_vars['codebook_loss'] = codebook_loss
            log_vars['@g_loss'] = g_loss

            if self.curr_iterations > self.gan_starts:
                log_vars['adversarial_loss'] = adversarial_loss
                log_vars['d_real'] = l_d_real
                log_vars['d_fake'] = l_d_fake
                log_vars['@d_loss'] = l_d_real + l_d_real
                self.ema_d_model.update()

            self.ema_g_model.update()

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
        self.log_codebook()
        evaluate_vq.evaluation(model=self.ema_g_model,
                               eval_data_loaders=self.eval_data_loaders,
                               epoch=epoch,
                               criterions=self.criterion,
                               writer=self.writer,
                               args=self.args,
                               logger=self.logger)

    @master_only
    def log_codebook(self, up_factor=2):
        codebook, num_code = self.model_no_ddp(self.g_model).vis_codebook(up_factor=up_factor)
        image = wandb.Image(codebook, caption=f"num_code-{num_code}")
        wandb.log({"Codebook": image})

    def save_model(self, path, *args):

        state_dict_saver(os.path.join(path, f"{self.model_no_ddp(self.g_model)}.pt"), self.ema_g_model.ema_model)
        state_dict_saver(os.path.join(path, f"{self.model_no_ddp(self.d_model)}.pt"), self.ema_d_model.ema_model)

    def save_ckpt(self, path, epoch):
        ckpt_saver(os.path.join(path, "latest.pt"),
                   g_model=self.ema_g_model.ema_model,
                   g_optimizer=self.g_optimizer,
                   g_scheduler=self.g_scheduler,
                   d_model=self.ema_d_model.ema_model,
                   d_optimizer=self.d_optimizer,
                   d_scheduler=self.d_scheduler,
                   epoch=epoch)
