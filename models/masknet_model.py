import os

import torch

from omegaconf import OmegaConf

import common
from arch.fema_vqgan_arch import FaceCoderNet
from utils.logger_utils import tb_writer, loss_printer
from .evaluation import evaluate_mask
from utils.train_utils import state_dict_saver, ckpt_saver

from .modules.masking import Masking
from .basic_model import BasicModel


class MaskNetModel(BasicModel):

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
        super().__init__(opt, total_iterations=opt.total_iterations)

        self.writer = writer

        self.local_rank = opt.local_rank

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.train_data_loader = train_data_loader
        self.eval_data_loaders = eval_data_loaders

        # load vqgan
        logger.info(f"Create vqvae from {opt.vq_config_path}")
        vq_config = OmegaConf.load(opt.vq_config_path)

        self.vqgan = FaceCoderNet(**vq_config.g_model).to(self.local_rank)
        assert os.path.exists(opt.vq_state_dict), opt.vq_state_dict
        self.vqgan.load_state_dict(torch.load(opt.vq_state_dict, map_location='cpu'))
        logger.info(f"Load vqvae weight from {opt.vq_state_dict}")
        self.vqgan.eval()
        for p in self.vqgan.parameters():
            p.requires_grad = False

        # load mask module
        self.mask = Masking(half_precision=True).to(self.local_rank)

        self.no_ddp_model = self.model_no_ddp(model)
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

            x, _ = batch

            bsz = x.size(0)

            x = x.to(self.local_rank, non_blocking=True)
            # y = y.to(self.local_rank, non_blocking=True)

            # mask face
            with torch.no_grad():
                # generate masked x
                masked_x = self.mask(x)
                # generate latent information for masked x
                latent_mx = self.vqgan.encode(masked_x)
                qmx, _, qmx_info = self.vqgan.quantize(z=latent_mx)
                qmx_indices = qmx_info['min_encoding_indices'].reshape(bsz, -1)

                # generate latent information for unmasked x
                latent_x = self.vqgan.encode(x)
                qx, _, qx_info = self.vqgan.quantize(z=latent_x)
                qx_indices = qx_info['min_encoding_indices'].reshape(bsz, -1)

                # generate mask ground truth -- indicate which semantic info is changed after masking applied
                # soft ground truth
                # gt_soft = vector_similarity(qmx.flatten(2), qx.flatten(2), dim=1)  # quantized vector
                # gt_soft = vector_similarity(latent_mx.flatten(2), latent_x.flatten(2), dim=1)  # un-quantized vector
                # gt_soft = torch.nn.functional.cosine_similarity(latent_mx.flatten(2), latent_x.flatten(2),
                #                                                 dim=1) / 2 + 0.5
                # hard ground truth
                gt_hard = 1 - qmx_indices.eq(qx_indices).float()

                # key_gt = gt_hard.nonzero(as_tuple=True)

            self.optimizer.zero_grad()

            masked_indicator = self.model(masked_x)

            # loss_soft = torch.nn.functional.mse_loss(masked_indicator, gt_soft)
            # loss_hard = torch.nn.functional.binary_cross_entropy(masked_indicator[key_gt], gt_hard[key_gt])
            loss_hard = torch.nn.functional.binary_cross_entropy(masked_indicator, gt_hard)

            loss = loss_hard

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            log_vars['@lr'] = self.scheduler.get_last_lr()[0]
            log_vars['@loss'] = loss
            log_vars['BCE'] = loss_hard
            # log_vars['Soft'] = loss_soft

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
        evaluate_mask.evaluation(model=self.model,
                                 eval_data_loaders=self.eval_data_loaders,
                                 epoch=epoch,
                                 criteria=self.criteria,
                                 writer=self.writer,
                                 args=self.opt,
                                 logger=self.logger,
                                 mask=self.mask,
                                 vqgan=self.vqgan)

    def save_model(self, path, *opt):
        state_dict_saver(os.path.join(path, f"{self.no_ddp_model}.pt"), self.no_ddp_model)

    def save_ckpt(self, path, epoch):
        ckpt_saver(os.path.join(path, "latest.pt"),
                   model=self.no_ddp_model,
                   optimizer=self.optimizer,
                   scheduler=self.scheduler,
                   epoch=epoch)


def vector_similarity(x: torch.Tensor, y: torch.Tensor, dim: int, norm: bool = False) -> torch.Tensor:
    """
    Compute the similarity in continuous manner
    """
    if norm:
        return torch.nn.functional.cosine_similarity(x, y, dim=dim) / 2 + 0.5
    else:
        return torch.nn.functional.cosine_similarity(x, y, dim=dim)
