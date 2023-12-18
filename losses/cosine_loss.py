import torch.nn as nn


class CosineLoss(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.loss = nn.BCELoss()
        self.cos_sim = nn.CosineSimilarity()

    def forward(self, a, v, y):
        distance = self.cos_sim(a, v)
        loss = self.loss(distance.unsqueeze(0), y)

        return loss * self.weight
