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


class Generator(nn.Module):

    def __init__(self, nc, nz, nef, norm):
        super(Generator, self).__init__()
        if norm == 'Nan':
            self.main = nn.Sequential(
                View((-1, nz, 1, 1)),
                # No normalization
                nn.ConvTranspose2d(nz, nef * 8, 4),
                nn.ReLU(),

                nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1),
                nn.ReLU(),

                nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1),
                nn.ReLU(),

                nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1),
                nn.ReLU(),

                nn.ConvTranspose2d(nef, nc, 4, 2, 1),
                nn.Tanh()
            )
        elif norm == 'SN':
            self.main = nn.Sequential(
                View((-1, nz, 1, 1)),
                # SN
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
        elif norm == 'BN':
            self.main = nn.Sequential(
                View((-1, nz, 1, 1)),
                # BN
                nn.ConvTranspose2d(nz, nef * 8, 4),
                nn.BatchNorm2d(nef * 8),
                nn.ReLU(),

                nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1),
                nn.BatchNorm2d(nef * 4),
                nn.ReLU(),

                nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1),
                nn.BatchNorm2d(nef * 2),
                nn.ReLU(),

                nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1),
                nn.BatchNorm2d(nef),
                nn.ReLU(),

                nn.ConvTranspose2d(nef, nc, 4, 2, 1),
                nn.Tanh()
            )
        else:
            self.main = nn.Sequential(
                View((-1, nz, 1, 1)),
                # SN and BN
                SpectralNorm(nn.ConvTranspose2d(nz, nef * 8, 4)),
                nn.BatchNorm2d(nef * 8),
                nn.ReLU(),

                SpectralNorm(nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1)),
                nn.BatchNorm2d(nef * 4),
                nn.ReLU(),

                SpectralNorm(nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1)),
                nn.BatchNorm2d(nef * 2),
                nn.ReLU(),

                SpectralNorm(nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1)),
                nn.BatchNorm2d(nef),
                nn.ReLU(),

                nn.ConvTranspose2d(nef, nc, 4, 2, 1),
                nn.Tanh()
            )

    def forward(self, z):
        R = self.main(z)
        return R


class GeneratorAAE(nn.Module):
    def __init__(self, nz, nef, nc):
        super(GeneratorAAE, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, nef * 16),
            View((-1, nef * 16, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 16, nef * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef * 2, nef, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(nef, nc, 4, 2, 1, bias=False),
            nn.ReLU(True)
        )

    def forward(self, input):
        R = self.main(input)
        return R