from functools import partial

import numpy as np

import torch
import torch.nn as nn

import scipy.stats as stats

from omegaconf import OmegaConf
from einops import rearrange

from utils.logging_tool import get_logger

from .mage_basic_arch import LabelSmoothingCrossEntropy, MlmLayer, Block, CrossBlock, BertEmbeddings
from .fema_vqgan_arch import FaceCoderNet
from .audionet_arch import AudioNet


class DoubleConditionedMAGE(nn.Module):
    """
        Masked Autoencoder with VisionTransformer backbone

        # for lip-sync work, extra CrossattentionBlock would be inserted into MAGE_Decoder.

        # Solution1:
        # wav2lip like way of info concatenation:
        # concate (reference float_latent(output from your pretrained VQGAN_Encoder), audioMel)
        # treat this chunk altogether into a single CrossattentionBlock

        # Solution2:
        # treat reference float_latent and audio mel seperately
        # two adjacent crossattention blocks are inserted, and two conditional info treated sequentially


        # here with the Solution1 version, but if required, we could easily formulate Solution2.

        # make sure you replace the import and initialization for our FACE_VQGAN

    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vq_config_path='config/vqgan.yml', vq_state_dict=None,
                 use_audio_reference=True, use_image_reference=True):
        super().__init__()
        logger = get_logger()
        # --------------------------------------------------------------------------
        # VQGAN specifics
        # replace this part with our face_vqgan specifications
        vq_config = OmegaConf.load(vq_config_path)
        self.vqgan = FaceCoderNet(**vq_config.g_model)
        if vq_state_dict:
            self.vqgan.load_state_dict(torch.load(vq_state_dict, map_location='cpu'))
            logger.info(f"Load vq model weight from {vq_state_dict}")

        self.vqgan_embed_dim = self.vqgan.embed_dim
        self.codebook_size = self.vqgan.codebook_size
        # [0, ..., codebook_size - 1, fake_class_label, mask_token_label]
        vocab_size = self.codebook_size + 1 + 1  # codebook size, 1 for mask token, 1 for fake_label
        self.fake_class_label = self.codebook_size
        self.mask_token_label = self.codebook_size + 1

        # froze the pretrained vqgan model
        for param in self.vqgan.parameters():
            param.requires_grad = False

        logger.info(f"MAGE_encoder_related_embeddingindex: "
                    f"Codebook Size: {self.codebook_size} "
                    f"Vocab Size: {vocab_size} "
                    f"Fake Class Label: {self.fake_class_label} "
                    f"Mask Token Label: {self.mask_token_label}")

        # create audio encoder based on decoder_embed_dim
        self.use_audio_reference = use_audio_reference
        if use_audio_reference:
            self.audio_net = AudioNet(emb_dim=decoder_embed_dim)

        # create image reference mapping that map img ref emb_dim to decoder_embed_dim
        self.use_image_reference = use_image_reference
        if use_image_reference:
            self.decoder_embed_mapping = nn.Linear(self.vqgan_embed_dim, decoder_embed_dim)

        logger.info(f"use_audio_reference:{use_audio_reference}")
        logger.info(f"use_image_reference:{use_image_reference}")

        # MAGE variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        # --------------------------------------------------------------------------
        # MAGE encoder specifics
        # patch_embed, cls_token, pos_embed is never used in MAGE Encoder
        # check whether need to apply pos_embed for Encoder, not sure about this
        dropout_rate = 0.1
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # num_patches = self.patch_embed.num_patches
        num_patches = self.vqgan.latent_resolution ** 2

        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=num_patches + 1,
                                        dropout=0.1)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
        #                               requires_grad=False)  # fixed sin-cos embedding

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

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        self.transformer_decoder = TransformerDecoder(decoder_embed_dim, decoder_num_heads, depth=decoder_depth,
                                                      mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None,
                                                      norm_layer=norm_layer, drop=dropout_rate, attn_drop=dropout_rate,
                                                      cross_attn=self.use_image_reference or self.use_audio_reference)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)

        self.norm_pix_loss = norm_pix_loss

        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
        #                                     cls_token=True)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
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

    @torch.no_grad()
    def encode_reference(self, ref):
        # encode reference image to latent feature
        z_ref = self.vqgan.encode(ref)
        z = rearrange(z_ref, 'b c h w -> b (h w) c').contiguous()  # reshape from bsz, c, h, w --> bsz, h*w, c
        z_map = self.decoder_embed_mapping(z)
        return z_map

    def forward_encoder(self, x, gt=None):
        # tokenization
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

            # if no gt then self-supervised learning, in this case x should be a complete image
            else:
                gt_indices = x_indices.clone().detach().long()

                # random masking
                bsz, seq_len = x_indices.size()
                mask_ratio_min = self.mask_ratio_min
                if self.training:
                    mask_rate = self.mask_ratio_generator.rvs(1)[0]
                else:
                    mask_rate = 0.75  # fix for testing
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
                        # print(noise.shape)
                    sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
                    cutoff_drop = sorted_noise[:, num_dropped_tokens - 1:num_dropped_tokens]
                    cutoff_mask = sorted_noise[:, num_masked_tokens - 1:num_masked_tokens]
                    token_drop_mask = noise.less_equal(cutoff_drop).float()
                    token_all_mask = noise.less_equal(cutoff_mask).float()
                    # if not self.training:
                    #     break
                    if token_drop_mask.sum() == bsz * num_dropped_tokens and token_all_mask.sum() == bsz * num_masked_tokens:
                        break
                    else:
                        print("Rerandom the noise!")

        print(mask_rate, num_dropped_tokens, num_masked_tokens, token_drop_mask.sum(dim=1), token_all_mask.sum(dim=1))
        x_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label
        # print("Masekd num token:", torch.sum(x_indices == self.mask_token_label, dim=1))

        # concate class token
        x_indices = torch.cat(
            [torch.zeros(x_indices.size(0), 1).to(device=x_indices.device), x_indices], dim=1)
        x_indices[:, 0] = self.fake_class_label
        token_drop_mask = torch.cat([torch.zeros(x_indices.size(0), 1).to(device=x_indices.device), token_drop_mask],
                                    dim=1)
        token_all_mask = torch.cat([torch.zeros(x_indices.size(0), 1).to(device=x_indices.device), token_all_mask],
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

        # apply Transformer blocks
        x = input_embeddings_after_drop
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

        if self.use_audio_reference and self.use_image_reference:
            assert audio_emb.size(-1) == ref_emb.size(-1)
            ref = torch.cat([audio_emb, ref_emb], dim=1)
        elif self.use_audio_reference:
            ref = audio_emb
        elif self.use_image_reference:
            ref = ref_emb
        else:
            ref = x
        # apply Transformer blocks
        x = self.transformer_decoder(x, ref, ref)

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
            # acc = 0
            # for pred_seq, gt_seq, mask_seq in zip(pred.reshape(bsz, seq_len), gt_indices, mask[:, 1:]):
            #     masked_idx = mask_seq.nonzero(as_tuple=True)
            #     acc += pred_seq[masked_idx].eq(gt_seq[masked_idx]).float().sum() / mask_seq.sum()
            return loss, acc

    def forward(self, imgs, gt=None, ref=None, audio=None, generate=False):
        # encoder
        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(imgs, gt)
        bsz = latent.size(0)
        # todo: generate audio embedding, reference embedding

        ref_emb = self.encode_reference(ref=ref) if self.use_image_reference else None
        audio_emb = self.audio_net(audio) if self.use_audio_reference else None
        # decoder
        logits = self.forward_decoder(latent, audio_emb=audio_emb, ref_emb=ref_emb,
                                      token_drop_mask=token_drop_mask, token_all_mask=token_all_mask)
        # compute prediction_masked_token_loss
        loss = self.forward_loss(gt_indices, logits, token_all_mask)
        if generate:
            latent_res = self.vqgan.latent_resolution
            vq_emb = self.vqgan_embed_dim
            logits = logits[:, 1:, :self.vqgan.codebook_size]
            _, pred_indices = torch.topk(logits, k=1)
            pred_indices = pred_indices * token_all_mask[:, 1:, None] + gt_indices[..., None] * (
                    1 - token_all_mask[:, 1:, None])
            z_q = self.vqgan.quantizer.get_codebook_entry(pred_indices.long(),
                                                          shape=(bsz, latent_res, latent_res, vq_emb))
            imgs = self.vqgan.decode(z_q)
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

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim, num_heads, depth, mlp_ratio, norm_layer,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., cross_attn=True):
        super().__init__()
        module = CrossBlock if cross_attn else Block
        self.cross_attn = cross_attn
        self.decoder_blocks = nn.ModuleList([
            module(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                   norm_layer=norm_layer, drop=drop, attn_drop=attn_drop) for _ in range(depth)])

    def forward(self, x, key, query):
        assert key.size() == query.size()
        for blk in self.decoder_blocks:
            x = blk(x, key, query) if self.cross_attn else blk(x)
        return x


def lip_mage_vit_base_patch16(**kwargs):
    model = DoubleConditionedMAGE(
        patch_size=32, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def lip_mage_vit_small(**kwargs):
    model = DoubleConditionedMAGE(
        patch_size=32, embed_dim=384, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def lip_mage_vit_tiny(**kwargs):
    model = DoubleConditionedMAGE(
        patch_size=32, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
