from functools import partial
import torch.nn as nn
import torch.nn.functional as F

from .auxiliary_arch import AudioAttNet, AudioNet
from .conditioned_mage_arch import DoubleConditionedMAGE


class SyncMage(DoubleConditionedMAGE):

    def __init__(self, num_embed=1024, **kwargs):
        super(SyncMage, self).__init__(**kwargs)
        for p in self.parameters():
            p.requires_grad = False

        self.img_probe = nn.Linear(self.encoder_embed_dim, num_embed)

        self.audio_encoder = AudioAttNet(seq_len=16, dim_aud=80)
        self.audio_probe = nn.Linear(80, num_embed)

    def forward(self, imgs, gt=None, ref=None, audio=None, generate=False, return_loss=True):
        # encode image with pre-trained Transformer encoder
        img_latent, _, _, _ = self.forward_encoder(imgs, gt)
        img_embed = self.img_probe(img_latent[:, 0]).squeeze(1)

        # encode audio with audio_encoder, audio input should be bsz, 1, 80, 16 (only for consistency with wav2lip)
        audio_latent = self.audio_encoder(audio)
        audio_embed = self.audio_probe(audio_latent).squeeze(1)

        assert audio_embed.size() == img_embed.size(), f"{audio_embed.size()}, {img_embed.size()}"

        return audio_embed, img_embed


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
