import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


# class VectorQuantizer(nn.Module):
#     """
#     see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
#     ____________________________________________
#     Discretization bottleneck part of the VQ-VAE.
#     Inputs:
#     - n_e : number of embeddings
#     - e_dim : dimension of embedding
#     - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
#     _____________________________________________
#     """
#
#     def __init__(self, n_e, e_dim, beta=0.25):
#         super().__init__()
#         self.n_e = int(n_e)  # codebook size
#         self.e_dim = int(e_dim)  # number of embeddings
#         self.beta = beta
#         self.embedding = nn.Embedding(self.n_e, self.e_dim)
#         self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
#
#     def dist(self, x, y):
#         return torch.sum(x ** 2, dim=1, keepdim=True) + \
#             torch.sum(y ** 2, dim=1) - 2 * \
#             torch.matmul(x, y.t())
#
#     def gram_loss(self, x, y):
#         b, h, w, c = x.shape
#         x = x.reshape(b, h * w, c)
#         y = y.reshape(b, h * w, c)
#
#         gmx = x.transpose(1, 2) @ x / (h * w)
#         gmy = y.transpose(1, 2) @ y / (h * w)
#
#         return (gmx - gmy).square().mean()
#
#     def forward(self, z, gt_indices=None):
#         """
#         Args:
#             z: input features to be quantized, z (continuous) -> z_q (discrete)
#                z.shape = (batch, channel, height, width)
#             gt_indices: feature map of given indices, used for visualization.
#         """
#         # reshape z -> (batch, height, width, channel) and flatten
#         z = z.permute(0, 2, 3, 1).contiguous()
#         z_flattened = z.view(-1, self.e_dim)
#
#         codebook = self.embedding.weight
#
#         d = self.dist(z_flattened, codebook)
#
#         # find closest encodings
#         min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
#         min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
#         min_encodings.scatter_(1, min_encoding_indices, 1)
#
#         if gt_indices is not None:
#             gt_indices = gt_indices.reshape(-1)
#
#             gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
#             gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
#             gt_min_onehot.scatter_(1, gt_min_indices, 1)
#
#             z_q_gt = torch.matmul(gt_min_onehot, codebook)
#             z_q_gt = z_q_gt.view(z.shape)
#
#         # get quantized latent vectors
#         z_q = torch.matmul(min_encodings, codebook)
#         z_q = z_q.view(z.shape)
#
#         e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
#         q_latent_loss = torch.mean((z_q - z.detach()) ** 2)
#
#         codebook_loss = q_latent_loss + e_latent_loss * self.beta
#
#         # preserve gradients
#         z_q = z + (z_q - z).detach()
#
#         # reshape back to match original input shape
#         z_q = z_q.permute(0, 3, 1, 2).contiguous()
#
#         return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])
#
#     def get_codebook_entry(self, indices):
#         b, _, h, w = indices.shape
#
#         indices = indices.flatten().to(self.embedding.weight.device)
#         min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
#         min_encodings.scatter_(1, indices[:, None], 1)
#
#         # get quantized latent vectors
#         z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
#         z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
#         return z_q

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight ** 2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        mean_distance = torch.mean(d)
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # min_encoding_scores, min_encoding_indices = torch.topk(d, 1, dim=1, largest=False)
        # [0-1], higher score, higher confidence
        # min_encoding_scores = torch.exp(-min_encoding_scores/10)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": mean_distance
        }

    def get_codebook_entry(self, indices, shape=None):
        # input indices: batch*token_num -> (batch*token_num)*1
        # shape: batch, height, width, channel
        b, _, h, w = indices.shape

        indices = indices.view(-1, 1).to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        else:
            z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, straight_through=False, kl_weight=5e-4, temp_init=1.0):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv2d(num_hiddens, codebook_size, 1)  # projects last encoder layer to quantized logits
        self.embedding = nn.Embedding(codebook_size, emb_dim)

    def forward(self, z):
        hard = self.straight_through if self.training else True

        logits = self.proj(z)

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)

        z_q = torch.einsum("b n h w, n d -> b d h w", soft_one_hot, self.embedding.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()
        min_encoding_indices = soft_one_hot.argmax(dim=1)

        return z_q, diff, {
            "min_encoding_indices": min_encoding_indices
        }

    def get_codebook_entry(self, indices, shape=None):
        # input indices: batch*token_num -> (batch*token_num)*1
        # shape: batch, height, width, channel
        b, _, h, w = indices.shape

        indices = indices.view(-1, 1).to(self.embedding.weight.device)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        else:
            z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return z_q


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
        }

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        if shape is None:
            b, _, h, w = indices.shape
            shape = (b, h, w, -1)
        indices = indices.to(self.embedding.weight.device)

        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
