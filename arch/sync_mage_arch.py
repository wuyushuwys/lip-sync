import os
from functools import partial

import torch
import torch.nn as nn

from einops import rearrange
from omegaconf import OmegaConf
from timm.layers import use_fused_attn

from utils.logging_tool import get_logger

from .mage_basic_arch import Block, BertEmbeddings
from .fema_vqgan_arch import FaceCoderNet
from .auxiliary_arch import AudioEncoder
from .ops import PatchEmbed, TimestepEmbedding, Timesteps
from .modules.masking import Masking


class SyncMAGE(nn.Module):
    """
        Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(
            self,
            # transformer encoder config
            embed_dim=1024, depth=24, num_heads=16,
            # attention config
            mlp_ratio=4., norm_layer=nn.LayerNorm,
            # vqgan config
            vq_config_path='config/vqgan.yml', vq_state_dict=None,
    ):
        super().__init__()

        logger = get_logger()
        # --------------------------------------------------------------------------
        # VQGAN with reference control specifics

        vq_config = OmegaConf.load(vq_config_path)
        self.vqgan = FaceCoderNet(**vq_config.g_model)
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
        logger.info(f"Sync Mage info:")
        logger.info(f"Codebook Size: {self.codebook_size}")
        logger.info(f"Vocab Size: {vocab_size}")
        logger.info(f"Fake Class Label: {self.fake_class_label}")
        logger.info(f"Mask Token Label: {self.mask_token_label}")

        # create audio encoder based on decoder_embed_dim
        self.audio_net = AudioEncoder(emb_dim=1024)

        # --------------------------------------------------------------------------
        # MAGE encoder specifics
        dropout_rate = 0.1
        num_patches = self.vqgan.latent_resolution ** 2

        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=num_patches + 1,
                                        dropout=0.1)

        self.patch_embed = PatchEmbed(height=256, width=256, patch_size=256 // 32, embed_dim=embed_dim, cls_token=True)

        self.spatio_temporal_encoder = TransformerSpatioTemporal(embed_dim, num_heads, depth=depth, mlp_ratio=mlp_ratio,
                                                                 qkv_bias=True, qk_scale=None,
                                                                 norm_layer=norm_layer, drop=dropout_rate,
                                                                 attn_drop=dropout_rate)

        self.norm = norm_layer(embed_dim)

        self.img_proj = nn.Linear(embed_dim, 1024)

        self.initialize_weights()

        self.mask_module = Masking(size=256, half_precision=True, norm=False)

    def initialize_weights(self):
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

    def add_class_token(self, x):
        x = torch.cat([torch.zeros(x.size(0), 1, device=x.device), x], dim=1)
        x[:, 0] = self.fake_class_label
        return x

    def index_generator(self, mask_x, x):

        bsz = mask_x.size(0)

        with torch.no_grad():
            # encode and quantize x
            z_x = self.vqgan.encode(mask_x)
            z_q_x, _, quantizer_info_x = self.vqgan.quantize(z_x)
            x_indices = quantizer_info_x['min_encoding_indices'].reshape(bsz, -1)

            # determine masked token based on z_q_indices difference
            z_gt = self.vqgan.encode(x)
            z_q_gt, _, quantizer_info_gt = self.vqgan.quantize(z_gt)
            gt_indices = quantizer_info_gt['min_encoding_indices'].reshape(bsz, -1)
            token_all_mask = x_indices.not_equal(gt_indices).float()

        x_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label

        # concate class token
        x_indices = self.add_class_token(x_indices)

        x_indices = x_indices.long()
        # bert embedding
        input_embeddings = self.token_emb(x_indices)

        return input_embeddings

    def forward_encoder(self, x):

        # add patch position embed
        x = self.patch_embed(x)

        # apply Transformer blocks
        x = self.spatio_temporal_encoder(x)
        x = self.norm(x)
        x = self.img_proj(x)
        return x

    def face_mask_index(self, x):
        with torch.no_grad():
            x = rearrange(x, 'b (t c) h w -> (b t) c h w', c=3)
            mask_x = self.mask_module(x, mask_face=False)

        return self.index_generator(mask_x, x)

    def forward(self, mel, images):

        # generate mask face embed
        x = self.face_mask_index(images)
        # encoder
        latent_x = self.forward_encoder(x).squeeze(1)

        latent_audio = self.audio_net(mel).squeeze(1)

        return latent_audio, latent_x

    def __str__(self):
        return self.__class__.__name__.lower()


class SpatioTemporalLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_ratio, norm_layer,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., num_frame=5):
        super().__init__()

        self.num_frame = num_frame

        time_embed_dim = embed_dim * 4
        self.time_proj = Timesteps(embed_dim, True, 0)

        self.time_pos_embed = TimestepEmbedding(embed_dim, time_embed_dim, out_dim=embed_dim)

        self.spatio_blocks = Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   norm_layer=norm_layer, drop=drop, attn_drop=attn_drop)

        self.temporal_blocks = Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     norm_layer=norm_layer, drop=drop, attn_drop=attn_drop)

        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        residual = x

        batch_frame, L, N = x.size()
        batch_size = batch_frame // self.num_frame

        num_frames_emb = torch.arange(self.num_frame, device=x.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        t_emb = t_emb.to(dtype=x.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # spatial input [batch * num_frames, H * W, N]
        x = self.spatio_blocks(x)

        # add time embedding
        x = x + emb

        # temporal input [batch * H * W, num_frames, N]
        x = rearrange(x, '(b t) l n-> (b l) t n', b=batch_size)
        x = self.temporal_blocks(x)
        x = rearrange(x, '(b l) t n-> (b t) l n', b=batch_size)

        x = self.proj_out(x)

        return x + residual


class TransformerSpatioTemporal(nn.Module):

    def __init__(self, embed_dim, num_heads, depth, mlp_ratio, norm_layer,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., num_frame=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            SpatioTemporalLayer(embed_dim, num_heads, mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                norm_layer=norm_layer, drop=drop,
                                attn_drop=attn_drop, num_frame=num_frame) for _ in range(depth)])

        self.num_frame = num_frame

        self.out_proj = nn.Linear(num_frame, 1)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        batch_size = x.size(0) // self.num_frame
        # [batch_size * frames, 1, num_embed]
        x = x[:, :1]
        # [batch_size, num_embed, frames]
        x = rearrange(x, '(b t) l n-> b n (t l)', b=batch_size)
        # [batch_size, num_embed, 1]
        x = self.out_proj(x).permute(0, 2, 1)

        return x


def sync_mage_vit_base(**kwargs):
    model = SyncMAGE(
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def sync_mage_vit_small(**kwargs):
    model = SyncMAGE(
        embed_dim=384, depth=6, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def sync_mage_vit_tiny(**kwargs):
    model = SyncMAGE(
        embed_dim=192, depth=4, num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
