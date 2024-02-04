import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, codebook_size, emb_dim, beta):
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
