import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import tensorboardX

from Critic import Discriminator
from Generator import Generator

x = torch.randn(20, 128)
netG = Generator(3, 128, 64,"BN")
netD = Discriminator()
writer = SummaryWriter('./test')
writer.add_graph(netG, x)
writer.close()
# grad = torch.randn(4,8)
# print(torch.sqrt(torch.sum(grad ** 2, dim=1)))
# print(grad.norm(2,dim=1))