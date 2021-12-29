import torch.nn as nn
import numpy as np


class DiscriminatorLi(nn.Module):
    def __init__(self,nc, img_size):
        super(DiscriminatorLi, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod((nc, img_size, img_size))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
