import numpy as np
import torch
from torch import nn
from spectralNorm import SpectralNorm


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class GeneratorDC(nn.Module):

    def __init__(self, nc, nz, nef):
        super(GeneratorDC, self).__init__()

        self.main = nn.Sequential(
            View((-1, nz, 1, 1)),

            SpectralNorm(nn.ConvTranspose2d(nz, nef * 8, 4)),
            nn.ReLU(),

            SpectralNorm(nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1)),
            nn.ReLU(),

            SpectralNorm(nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1)),
            nn.ReLU(),

            SpectralNorm(nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1)),
            nn.ReLU(),

            nn.ConvTranspose2d(nef, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        R = self.main(z)
        return R
