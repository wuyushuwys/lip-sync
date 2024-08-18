import os
from functools import partial

import numpy as np

import torch
import torch.nn as nn

import scipy.stats as stats

from omegaconf import OmegaConf
from einops import rearrange
from timm.layers import use_fused_attn

from utils.logging_tool import get_logger

from .mage_basic_arch import MlmLayer, Block, CrossBlock, BertEmbeddings, LabelSmoothingCrossEntropy
from .fema_temporal_vqgan_arch import FaceCoderTemporalNet
from .ref_control_net_arch import RefControlNet
from .auxiliary_arch import AudioNet, AudioEncoder, AudioPretrainedEncoder
from .ops import PositionalEncoding, get_2d_sincos_pos_embed


class DoubleTemporalConditionedMAGE(nn.Module):
    """
        Masked Autoencoder with VisionTransformer backbone
        for lip-sync work, extra CrossAttentionBlock would be inserted into MAGE_Decoder.
    """

    def __init__(
            self,
            # some image related arguments. not in use
            img_size=256, patch_size=16, in_chans=3,
            # transformer encoder config
            embed_dim=1024, depth=24, num_heads=16,
            # transformer decoder config
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            # attention config
            mlp_ratio=4., norm_layer=nn.LayerNorm,
            # pixel level loss related
            norm_pix_loss=False, gumble_softmax=False,
            # mask modeling related
            mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25, eval_mask_ratio=0.5,
            # vqgan config
            vq_config_path='config/vqgan.yml', vq_state_dict=None,
            mage_pretrain_ckpt_path=None,
            # reference information config
            use_audio_reference=True, audio_weight_path=None, num_audio_embed=1024, modulate_type='all',
            use_image_reference=True,
            tokenize_reference=False,
            # reference control model config
            ref_control=False, ref_controller_state_dict=None, ref_control_adaptive=False
    ):
        super().__init__()

        assert modulate_type in ['msa', 'mlp', 'all'], f"got unexpected modulation {modulate_type}"

        logger = get_logger()
        # --------------------------------------------------------------------------
        # VQGAN with reference control specifics
        self.ref_control = ref_control
        if ref_control:
            self.ref_controller = RefControlNet(vq_config_path=vq_config_path,
                                                vq_state_dict=vq_state_dict)
            if ref_controller_state_dict:
                self.ref_controller.load_state_dict(torch.load(ref_controller_state_dict, map_location='cpu'))
                logger.info(f"Enable reference control: {ref_control}")
            else:
                logger.warning("No ref_controller_state_dict")
            setattr(self, 'vqgan', self.ref_controller.vqgan)

            # frozen the pretrained ref_controller model
            logger.info("Frozen reference controller")
            for p in self.ref_controller.parameters():
                p.requires_grad = False

        else:
            # VQGAN specifics
            vq_config = OmegaConf.load(vq_config_path)
            self.vqgan = FaceCoderTemporalNet(**vq_config.g_model)
            if vq_state_dict is not None:
                if os.path.exists(vq_state_dict):
                    self.vqgan.load_state_dict(torch.load(vq_state_dict, map_location='cpu'))
                    logger.info(f"Load vq model weight from {vq_state_dict}")
                else:
                    raise FileNotFoundError(f"vq_model weight {vq_state_dict} not found")
            else:
                logger.info(f"Not pretrain vq model weight provided")

            # froze the pretrained vqgan model
            for p in self.vqgan.parameters():
                p.requires_grad = False

        self.vqgan_embed_dim = self.vqgan.embed_dim
        self.codebook_size = self.vqgan.codebook_size
        # [0, ..., codebook_size - 1, fake_class_label, mask_token_label]
        vocab_size = self.codebook_size + 1 + 1  # codebook size, 1 for mask token, 1 for fake_label
        self.fake_class_label = self.codebook_size  # fake token is said to gather global information among all tokens
        self.mask_token_label = self.codebook_size + 1
        self.encoder_embed_dim = embed_dim

        logger.info(f"Use Flash Attention: {use_fused_attn()}")
        logger.info(f"MAGE_encoder_related_embeddingindex: "
                    f"Codebook Size: {self.codebook_size} "
                    f"Vocab Size: {vocab_size} "
                    f"Fake Class Label: {self.fake_class_label} "
                    f"Mask Token Label: {self.mask_token_label}")

        # create audio encoder based on decoder_embed_dim
        self.use_audio_reference = use_audio_reference
        if use_audio_reference:
            # self.audio_net = AudioEncoder(emb_dim=1024)
            self.audio_net = AudioPretrainedEncoder(audio_weight_path=audio_weight_path)

        # create image reference mapping that map img ref emb_dim to decoder_embed_dim
        self.use_image_reference = use_image_reference
        if use_image_reference:
            # if not tokenize_reference:
            self.decoder_embed_mapping = nn.Linear(self.vqgan_embed_dim, decoder_embed_dim, bias=True)

        self.tokenize_reference = tokenize_reference

        logger.info(f"use_audio_reference:{use_audio_reference}")
        logger.info(f"use_image_reference:{use_image_reference}")
        logger.info(f"tokenize_reference:{tokenize_reference}")

        # MAGE variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)
        self.eval_mask_ratio = eval_mask_ratio

        # --------------------------------------------------------------------------
        # MAGE encoder specifics
        dropout_rate = 0.1
        num_patches = self.vqgan.latent_resolution ** 2

        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=num_patches + 1,
                                        dropout=0.1)

        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, depth=depth, mlp_ratio=mlp_ratio,
                                                      qkv_bias=True, qk_scale=None,
                                                      norm_layer=norm_layer, drop=dropout_rate, attn_drop=dropout_rate)
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAGE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.pad_with_cls_token = True

        if not self.pad_with_cls_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        image_embed_dim = embed_dim if self.tokenize_reference else self.vqgan_embed_dim

        self.transformer_decoder = TransformerDecoder(
            decoder_embed_dim, decoder_num_heads, depth=decoder_depth,
            mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
            norm_layer=norm_layer, drop=dropout_rate, attn_drop=dropout_rate,
            cross_attn=self.use_image_reference,  # add information in cross attention
            modulation=self.use_audio_reference,  # add information in modulation
            audio_dim=num_audio_embed if use_audio_reference else None,  # embedding for audio
            img_dim=image_embed_dim if use_image_reference else None,
            modulate_type=modulate_type,
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)

        self.norm_pix_loss = norm_pix_loss
        self.gumble_softmax = gumble_softmax

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
        # self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        if not mage_pretrain_ckpt_path:
            self.initialize_weights()
        else:
            logger.info(f"Load pretrain weight from {mage_pretrain_ckpt_path}")
            mage_pretrain_weight = torch.load(mage_pretrain_ckpt_path, map_location='cpu')
            incompatible_keys = self.load_state_dict(mage_pretrain_weight, strict=False)
            assert len(
                incompatible_keys.unexpected_keys) == 0, f"do not expected unexpected_keys {incompatible_keys.unexpected_keys}"
            # logger.info(incompatible_keys)

            for name, p in self.named_parameters():
                # first, we only unfreeze new parameters
                if name in incompatible_keys.missing_keys:
                    continue
                # then, we unfreeze mlp parameters in decoder
                elif 'decoder_blocks' in name and 'mlp' in name:
                    continue
                p.requires_grad = False

        # unfreeze decoder_norm layer
        for p in self.decoder_norm.parameters():
            p.requires_grad = True

        # unfreeze mlm layer
        for p in self.mlm_layer.parameters():
            p.requires_grad = True

        if ref_control_adaptive:
            # only train controller
            for p in self.parameters():
                p.requires_grad = False
            for p in self.ref_controller.controller.parameters():
                p.requires_grad = True

    def reload_controller(self, state_dict_path):
        logger = get_logger()
        if hasattr(self, 'ref_controller'):
            logger.info(f"Load control weight from {state_dict_path}")
            self.ref_controller.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

    def initialize_weights(self):
        # initialization
        if not self.pad_with_cls_token:
            torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode_reference(self, ref, tokenize=False):
        # encode reference image to latent feature
        with torch.no_grad():
            z_ref = self.vqgan.encode(ref)
            # if tokenize:
            #     z_q_ref, _, quantizer_info_ref = self.vqgan.quantize(z_ref)
            #     ref_indices = quantizer_info_ref['min_encoding_indices'].reshape(ref.size(0), -1)
            #     # z = self.token_emb(self.add_class_token(ref_indices).long())  # we concat class token too
            #     z = self.token_emb(ref_indices.long())
            #     # z = self.decoder_embed(z)  # unified mapping
            # else:
            z = rearrange(z_ref, 'b c h w -> b (h w) c').contiguous()  # reshape bsz, c, h, w -> bsz, (h w), c
            z = self.decoder_embed_mapping(z)
        return z

    def add_class_token(self, x):
        x = torch.cat(
            [torch.zeros(x.size(0), 1, device=x.device), x], dim=1)
        x[:, 0] = self.fake_class_label
        return x

    def index_generator(self, x, gt=None):
        bsz = x.size(0)

        with torch.no_grad():
            # encode and quantize x
            z_x = self.vqgan.encode(x)
            z_q_x, _, quantizer_info_x = self.vqgan.quantize(z_x)
            x_indices = quantizer_info_x['min_encoding_indices'].reshape(bsz, -1)

            # determine masked token based on z_q_indices difference
            if gt is not None:
                z_gt = self.vqgan.encode(gt)
                z_q_gt, _, quantizer_info_gt = self.vqgan.quantize(z_gt)
                gt_indices = quantizer_info_gt['min_encoding_indices'].reshape(bsz, -1)
                token_all_mask = x_indices.not_equal(gt_indices).float()
                token_drop_mask = torch.zeros_like(token_all_mask)
            # if no gt then self-supervised learning, in this case x should be a complete image
            else:
                gt_indices = x_indices.clone().detach().long()

                # random masking
                bsz, seq_len = x_indices.size()
                mask_ratio_min = self.mask_ratio_min if self.training else 0
                if self.training:
                    mask_rate = self.mask_ratio_generator.rvs(1)[0]
                else:
                    mask_rate = self.eval_mask_ratio  # fix for testing
                num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
                num_masked_tokens = int(np.ceil(seq_len * mask_rate))

                # it is possible that two elements of the noise is the same, so do a while loop to avoid it

                while True:
                    if self.training:
                        noise = torch.rand(bsz, seq_len, device=x.device)  # noise in [0, 1]
                    else:
                        # generate fake noise for evaluation
                        num_subseq = seq_len // 4  # typically seq_len is a multiple of 4
                        fake_noise = torch.arange(0, 4, device=x.device)[None, :] / 4
                        noise = fake_noise.repeat(bsz, num_subseq)

                    sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
                    if self.training:
                        cutoff_drop = sorted_noise[:, num_dropped_tokens - 1:num_dropped_tokens]
                        token_drop_mask = noise.less_equal(cutoff_drop).float()
                    else:
                        token_drop_mask = torch.zeros_like(noise)
                    cutoff_mask = sorted_noise[:, num_masked_tokens - 1:num_masked_tokens]
                    token_all_mask = noise.less_equal(cutoff_mask).float()

                    if token_drop_mask.sum() == bsz * num_dropped_tokens and token_all_mask.sum() == bsz * num_masked_tokens:
                        break
                    else:
                        print("Rerandom the noise!")

        # print(mask_rate, num_dropped_tokens, num_masked_tokens, token_drop_mask.sum(dim=1), token_all_mask.sum(dim=1))
        x_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label
        # print("Masekd num token:", torch.sum(x_indices == self.mask_token_label, dim=1))

        # concate class token
        x_indices = self.add_class_token(x_indices)

        token_drop_mask = torch.cat([torch.zeros(x_indices.size(0), 1, device=x_indices.device), token_drop_mask],
                                    dim=1)
        token_all_mask = torch.cat([torch.zeros(x_indices.size(0), 1, device=x_indices.device), token_all_mask],
                                   dim=1)
        x_indices = x_indices.long()
        # bert embedding
        input_embeddings = self.token_emb(x_indices)
        # print("Input embedding shape:", input_embeddings.shape)
        bsz, seq_len, emb_dim = input_embeddings.shape

        # dropping
        token_keep_mask = 1 - token_drop_mask
        input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)
        # print("Input embedding after drop shape:", input_embeddings_after_drop.shape)

        return input_embeddings_after_drop, gt_indices, token_drop_mask, token_all_mask

    def forward_encoder(self, x, gt=None, return_attn=False):

        input_embeddings_after_drop, gt_indices, token_drop_mask, token_all_mask = self.index_generator(x, gt)

        # apply Transformer blocks
        x = input_embeddings_after_drop
        if return_attn:
            x, attn = self.transformer_encoder(x, return_attn)
            x = self.norm(x), attn
        else:
            x = self.transformer_encoder(x)
            x = self.norm(x)
        # print("Encoder representation shape:", x.shape)

        return x, gt_indices, token_drop_mask, token_all_mask

    def forward_decoder(self, x, audio_emb, ref_emb, token_drop_mask, token_all_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        if self.pad_with_cls_token:
            mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(token_all_mask.shape[0], token_all_mask.shape[1], 1)

        # put undropped tokens into original sequence
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # set undropped but masked positions with mask
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        # add pos embed
        x = x_after_pad + self.decoder_pos_embed_learned

        if self.use_image_reference:
            ref = ref_emb
        else:
            ref = x
        # apply Transformer blocks
        x = self.transformer_decoder(x, kv=ref, cond=audio_emb if self.use_audio_reference else None)

        x = self.decoder_norm(x)

        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        # print("Logits shape:", x.shape)

        return x

    def forward_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        loss = self.criterion(logits[:, 1:, :self.codebook_size].reshape(bsz * seq_len, -1),
                              gt_indices.reshape(bsz * seq_len))
        loss = loss.reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()  # mean loss on removed patches
        if self.training:
            return loss
        else:
            _, pred = torch.topk(logits[:, 1:, :self.codebook_size].reshape(bsz * seq_len, -1), k=1)
            acc = pred.flatten()[mask[:, 1:].flatten().nonzero(as_tuple=True)].eq(
                gt_indices.flatten()[mask[:, 1:].flatten().nonzero(as_tuple=True)]).sum() / mask[:, 1:].sum()
            return loss, acc

    def forward(self, imgs, gt=None, ref=None, audio=None, generate=False, num_batch=1, return_loss=True):
        # encoder
        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(imgs, gt)
        bsz = latent.size(0)

        # generate ref_emb and audio_emb when use_image/audio_reference
        ref_emb = self.encode_reference(ref=ref, tokenize=self.tokenize_reference) if self.use_image_reference else None
        audio_emb = self.audio_net(audio) if self.use_audio_reference else None

        # decoder
        logits = self.forward_decoder(latent, audio_emb=audio_emb, ref_emb=ref_emb,
                                      token_drop_mask=token_drop_mask, token_all_mask=token_all_mask)

        # compute prediction_masked_token_loss
        loss = self.forward_loss(gt_indices, logits, token_all_mask) if return_loss else None

        if generate:
            latent_res = self.vqgan.latent_resolution
            vq_emb = self.vqgan_embed_dim
            logits = logits[:, 1:, :self.vqgan.codebook_size]

            _, pred_indices = torch.topk(logits, k=1)
            pred_indices = pred_indices * token_all_mask[:, 1:, None] + gt_indices[..., None] * (
                    1 - token_all_mask[:, 1:, None])
            z_q = self.vqgan.quantizer.get_codebook_entry(pred_indices.long(),
                                                          shape=(bsz, latent_res, latent_res, vq_emb))

            imgs = self.vqgan.decode(z_q, num_batch=num_batch)

        return loss, imgs, token_all_mask

    def __str__(self):
        return self.__class__.__name__.lower()


class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, depth, mlp_ratio, norm_layer,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  norm_layer=norm_layer, drop=drop, attn_drop=attn_drop) for _ in range(depth)])

    def forward(self, x, return_attn=False):
        for idx, blk in enumerate(self.blocks, start=1):
            if return_attn and idx == len(self.blocks):
                x, attn = blk(x, return_attn)
                return x, attn
            else:
                x = blk(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim, num_heads, depth, mlp_ratio, norm_layer,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., cross_attn=True,
                 modulation=False, modulate_type='msa', audio_dim=None, img_dim=None):
        super().__init__()
        module = partial(CrossBlock if cross_attn else Block, modulation=modulation, modulate_type=modulate_type)
        self.cross_attn = cross_attn
        self.decoder_blocks = nn.ModuleList([
            module(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                   norm_layer=norm_layer, drop=drop, attn_drop=attn_drop) for _ in range(depth)])

        self.proj_audio = nn.ModuleList([
            nn.Sequential(
                nn.Linear(audio_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU(),
            ) for _ in range(depth)]) if audio_dim else None

        # self.proj_img = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(img_dim, embed_dim),
        #         nn.LayerNorm(embed_dim),
        #     ) for _ in range(depth)]) if img_dim else None

    def forward(self, x, kv, cond=None):

        for i, blk in enumerate(self.decoder_blocks):
            proj_audio = self.proj_audio[i](cond) if self.proj_audio is not None else None
            # kv = self.proj_img[i](kv) if self.proj_img is not None else None
            if self.cross_attn:
                x = blk(x, kv, proj_audio)
            else:
                x = blk(x, proj_audio)
        return x


def lip_mage_vit_base(**kwargs):
    model = DoubleTemporalConditionedMAGE(
        patch_size=32, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def lip_mage_vit_small(**kwargs):
    model = DoubleTemporalConditionedMAGE(
        patch_size=32, embed_dim=384, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def lip_mage_vit_tiny(**kwargs):
    model = DoubleTemporalConditionedMAGE(
        patch_size=32, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
