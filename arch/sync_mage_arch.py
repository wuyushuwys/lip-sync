from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from utils.logging_tool import get_logger

from .ops import PositionalEncoding
from .auxiliary_arch import AudioAttNet, AudioNet, AudioEncoder, LipAttn, LipEncoder
from .conditioned_mage_arch import DoubleConditionedMAGE, TransformerEncoder


class SyncMage(DoubleConditionedMAGE):

    def __init__(self, num_embed=1024, **kwargs):
        super(SyncMage, self).__init__(**kwargs)
        # freeze all parameters in pretrained vit
        for p in self.parameters():
            p.requires_grad = False
        # self.logger = get_logger()
        # unfreeze the last transformer block in vit
        # for p in self.transformer_encoder.parameters():
        #     p.requires_grad = True

        self.temporal_emb = PositionalEncoding(d_model=self.encoder_embed_dim)

        self.lip_encoder = TransformerEncoder(embed_dim=self.encoder_embed_dim,
                                              num_heads=12,
                                              depth=4,
                                              mlp_ratio=4,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.lip_proj = nn.Linear(self.encoder_embed_dim, num_embed)

        self.audio_encoder = AudioEncoder(emb_dim=num_embed)

    def forward(self, masked_img, unmasked_img=None, ref=None, audio=None, generate=False, return_loss=True):
        # reshape images
        masked_img = rearrange(masked_img, 'b (t c) h w -> (b t) c h w ', c=3)
        unmasked_img = rearrange(unmasked_img, 'b (t c) h w -> (b t) c h w ', c=3)

        # encode image with pre-trained Transformer encoder
        with torch.no_grad():
            img_latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(masked_img, gt=unmasked_img)

        # verify the valid batch with mask (we expect there will be mouth detected)
        # if no mouth detected (masked_img == unmasked_img), we drop this image.

        # generate batch drop mask
        # check whether image is masked -> reshape to 5-frame base -> check if non-mask frame in 5-frame set
        drop_batch = token_all_mask.sum(dim=-1).view(-1, 5).eq(0).sum(-1)
        preserved_batch = ~drop_batch

        # batch dropping
        batched_img_latent = rearrange(img_latent, '(b t) n d -> b t n d', t=5)  # 5-frame basis
        preserved_img_latent = batched_img_latent[preserved_batch]
        preserved_audio = audio[preserved_batch]

        # 5 frames processing
        img_latent_seq = preserved_img_latent[:, :, 0]
        img_latent_seq += self.temporal_emb(img_latent_seq)
        img_embed = self.lip_encoder(img_latent_seq)
        img_embed = self.lip_proj(img_embed[:, -1])

        # encode audio with audio_encoder, audio input should be bsz, 1, 80, 16 (only for consistency with wav2lip)
        audio_embed = self.audio_encoder(audio)

        assert audio_embed.size() == img_embed.size(), f"{audio_embed.size()}, {img_embed.size()}"

        audio_embed = F.normalize(audio_embed, p=2, dim=1)
        img_embed = F.normalize(img_embed, p=2, dim=1)

        return audio_embed, img_embed

    def __str__(self):
        return self.__class__.__name__.lower()


def sync_mage_vit_base(**kwargs):
    model = SyncMage(
        patch_size=32, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def sync_mage_vit_small(**kwargs):
    model = SyncMage(
        patch_size=32, embed_dim=384, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def sync_mage_vit_tiny(**kwargs):
    model = SyncMage(
        patch_size=32, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
