import torch
import torch.nn as nn

from models.syncnet import SyncNet


class CosineLoss(nn.Module):

    def __init__(self, loss_weight=1):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss = nn.BCELoss()
        self.cos_sim = nn.CosineSimilarity()

    def forward(self, a, v, y):
        assert a.size() == v.size()
        distance = self.cos_sim(a, v)
        loss = self.loss(distance.unsqueeze(1), y)

        return loss * self.loss_weight


class SyncLoss(nn.Module):

    def __init__(self, ckpt_path, loss_weight=1, window_size=5):
        super().__init__()

        self.expert_model = SyncNet()
        self.expert_model.load_state_dict(torch.load(ckpt_path))
        self.loss_weight = loss_weight
        self.window_size = window_size

        self.criterion = CosineLoss()

        self.expert_model.eval()

    def forward(self, mel: torch.Tensor, pred_y: torch.Tensor):
        pred_y = pred_y[..., pred_y.size(3) // 2:, :]
        pred_y = torch.cat(pred_y.unbind(dim=2), dim=1)
        # B, 3 * T, H//2, W
        a, v = self.expert_model(mel, pred_y)
        label = torch.ones(pred_y.size(0), 1).to(mel.device)
        return self.criterion(a, v, label)
