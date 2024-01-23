import torch

from timm.models.vision_transformer import PatchEmbed, Block

from torch import nn
from torch.nn import functional as F

from .ops import Conv2d, ResBlock, Shape, Conv2dNew, get_2d_sincos_pos_embed


class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),
            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0, act='relu'),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act='relu'),

            nn.AdaptiveAvgPool2d(1)
        )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, act='relu'),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act='relu'), )

        self._init_weights()

    def forward(self, audio_sequences, face_sequences):  # audio_sequences := (B, dim, T)
        # print(audio_sequences.shape, face_sequences.shape)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)

    def __str__(self):
        return "syncnet"


class SyncNetWhole(nn.Module):
    def __init__(self):
        super(SyncNetWhole, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),
            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0, act='relu'),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act='relu'),

            nn.AdaptiveAvgPool2d(1)
        )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, act='relu'),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act='relu'), )

        self._init_weights()

    def forward(self, audio_sequences, face_sequences):  # audio_sequences := (B, dim, T)
        # print(audio_sequences.shape, face_sequences.shape)

        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)

    def __str__(self):
        return "syncnet_CNN"


# class SyncNetWhole(nn.Module):
#     def __init__(self):
#         super(SyncNetWhole, self).__init__()
#
#         self.face_encoder = FaceSyncEncoder(img_size=256, patch_size=16,
#                                             in_chans=15, embed_dim=1024,
#                                             depth=4)
#
#         self.audio_encoder = nn.Sequential(
#             Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
#
#             Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
#             Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
#
#             Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#
#             Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
#             Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#
#             Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
#             Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
#
#             Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, act='relu'),
#             Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, act='relu'), )
#
#         self._init_weights()
#
#     def forward(self, audio_sequences, face_sequences):  # audio_sequences := (B, dim, T)
#         # print(audio_sequences.shape, face_sequences.shape)
#
#         face_embedding = self.face_encoder(face_sequences)
#         audio_embedding = self.audio_encoder(audio_sequences)
#
#         audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
#         face_embedding = face_embedding.view(face_embedding.size(0), -1)
#
#         audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
#         face_embedding = F.normalize(face_embedding, p=2, dim=1)
#
#         return audio_embedding, face_embedding
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def __str__(self):
#         return "syncnet_transformer"


class FaceSyncEncoder(nn.Module):

    def __init__(self, img_size=256, patch_size=16, patch_bias=False,
                 in_chans=15,
                 embed_dim=1024, depth=4, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, global_pool=True):
        super(FaceSyncEncoder, self).__init__()
        self.global_pool = global_pool
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, bias=patch_bias)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, qk_norm=False, norm_layer=norm_layer) for _ in range(depth)
        ])
        if self.global_pool:
            self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        else:
            self.norm = norm_layer(embed_dim)

        self.act = nn.ReLU(True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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

    def patchify(self, imgs):
        """
        imgs: (N, 15, H, W)
        x: (N, L, patch_size**2 *3)
        """
        bsz, ch, H, W = imgs.size()
        p = self.patch_embed.patch_size[0]
        assert H == W and H % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], ch, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(bsz, h * w, p ** 2 * ch))
        return x

    def random_masking(self, x, mask_ratio=0.75, descending=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        # with torch.no_grad():
        score = (x.mean(dim=-1) == 0).float()
        # sort noise for each sample
        ids_shuffle = torch.argsort(score, dim=1, descending=descending)  # ascend: remove is keep, large is keep

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked

    def forward(self, x, mask_ratio=0.75):

        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1, :]

        x = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return self.act(outcome)
