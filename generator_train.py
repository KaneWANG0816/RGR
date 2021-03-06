from __future__ import print_function
import argparse
import os
import random
from Generator import Generator
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from Critic import Discriminator  # Discriminator: D
from torch.utils.data import DataLoader
from loadDataset import TrainDataset
from tensorboardX import SummaryWriter
from math import ceil
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./rain100L/train/small/rain", help='path of input')
parser.add_argument("--gt_path", type=str, default="./rain100L/train/small/norain", help='path of gt')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=20, help='batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='Number of image channels')
parser.add_argument('--nz', type=int, default=128, help='size of noise z')
parser.add_argument('--nef', type=int, default=32, help='channel for Generator')
parser.add_argument('--ndf', type=int, default=64, help='channel for Critic')

parser.add_argument('--niter', type=int, default=100, help='number of iteration')
parser.add_argument('--resume', type=int, default=33, help='resume')
parser.add_argument('--lambda_gp', type=float, default=10, help='penalty coefficient of WGAN-GP')
parser.add_argument("--stage", type=int, default=[350, 450, 500, 550], help="learning rate adjust")

parser.add_argument('--lrD', type=float, default=0.0004, help='learning rate of Critic')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate of Generator')
parser.add_argument('--n_dis', type=int, default=5, help='critic iterations')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')

parser.add_argument('--log_dir', default='./log/BN/', help='path of logs')
parser.add_argument('--model_dir', default='./model/syn100lmodels_BN/', help='path of model')
parser.add_argument('--manualSeed', type=int, help='seed')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

try:
    os.makedirs(opt.log_dir)
except OSError:
    pass
try:
    os.makedirs(opt.model_dir)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True


def train_model(netG, netD, datasets, optimizerG, lr_schedulerG, optimizerD, lr_schedulerD, resume):
    data_loader = DataLoader(datasets, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers),
                             pin_memory=True)
    num_data = len(datasets)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = resume * 3000
    for epoch in range(opt.resume, opt.niter):
        tic = time.time()
        # train stage
        lrG = optimizerG.param_groups[0]['lr']
        lrD = optimizerD.param_groups[0]['lr']
        print('lr_G %f' % lrG)
        print('lrD %f' % lrD)
        for ii, data in enumerate(data_loader):
            input, gt = [x.cuda() for x in data]

            # Update Discriminator D
            # train with original data
            netD.train()
            netD.zero_grad()
            d_out_real, _, _ = netD(input)
            d_loss_real = - torch.mean(d_out_real)

            # Noise
            noise = torch.randn(opt.batchSize, opt.nz).cuda()
            rain_make = netG(noise)

            # train with fake
            # rain_make, mu_z, logvar_z, _ = netG(gt)

            input_fake = gt + rain_make
            d_out_fake, _, _ = netD(input_fake.detach())
            d_loss_fake = d_out_fake.mean()

            # Compute gradient penalty
            alpha = torch.rand(input.size(0), 1, 1, 1).cuda().expand_as(input)
            interpolated = Variable(alpha * input.data + (1 - alpha) * input_fake.data, requires_grad=True)
            out, _, _ = netD(interpolated)
            grad = torch.autograd.grad(outputs=out,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
            # Backward + Optimize
            errD = d_loss_real + d_loss_fake + opt.lambda_gp * d_loss_gp
            errD.backward()
            optimizerD.step()

            # Update G network
            if step % opt.n_dis == 0:
                # arr = rain_make.cpu().detach().numpy()
                # arr_ = arr[0].squeeze()
                # arr_ = arr_.swapaxes(0, 2)
                # cv2.imshow('image', arr_)
                # cv2.waitKey(1000)
                netG.train()
                netG.zero_grad()
                g_out_fake, _, _ = netD(input_fake)
                g_loss_fake = - g_out_fake.mean()
                errG = g_loss_fake
                errG.backward()
                optimizerG.step()
            if ii % 200 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:0>5d}/{:0>5d}, d_loss_fake={:5.2e} d_loss_real={:5.2e} lossgp={: ' \
                           '5.2e} errD={:5.2e} errG={:5.2e} '
                print(template.format(epoch + 1, opt.niter, ii, num_iter_epoch, d_loss_fake.item(), d_loss_real.item(),
                                      d_loss_gp.item(), errD.item(), errG.item()))

                writer.add_scalar('Dloss', errD.item(), step)
                writer.add_scalar('Gloss', errG.item(), step)
                writer.add_scalar('drloss', d_loss_real.item(), step)
                writer.add_scalar('dfloss', d_loss_fake.item(), step)
                writer.add_scalar('gploss', opt.lambda_gp * d_loss_gp.item(), step)
                x2 = vutils.make_grid(gt, normalize=True, scale_each=True)
                writer.add_image('Ground truth', x2, step)
                x3 = vutils.make_grid(input, normalize=True, scale_each=True)
                writer.add_image('Input image', x3, step)
                x5 = vutils.make_grid(input_fake.data, normalize=True, scale_each=True)
                writer.add_image('Generated image', x5, step)
                x6 = vutils.make_grid(rain_make.data, normalize=True, scale_each=True)
                writer.add_image('Rain Layer', x6, step)
            step += 1

        lr_schedulerG.step()
        lr_schedulerD.step()
        # save model

        save_path_model = os.path.join(opt.model_dir, 'G_state_' + str(epoch + 1) + '.pt')
        torch.save(netG.state_dict(), save_path_model)
        save_path_model = os.path.join(opt.model_dir, 'D_state_' + str(epoch + 1) + '.pt')
        torch.save(netD.state_dict(), save_path_model)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))
        print('-' * 100)
    writer.close()
    print('Reach the maximal epochs! Finish training')


def main(norm):
    netG = Generator(opt.nc, opt.nz, opt.nef, norm).cuda()
    netD = Discriminator(image_size=opt.imageSize, conv_dim=opt.ndf).cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD)

    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, opt.stage, gamma=0.5)
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, opt.stage, gamma=0.5)

    for _ in range(opt.resume):
        schedulerG.step()
        schedulerD.step()

    if opt.resume:
        netG.load_state_dict(torch.load(os.path.join(opt.model_dir, 'G_state_' + str(opt.resume) + '.pt')))
        netD.load_state_dict(torch.load(os.path.join(opt.model_dir, 'D_state_' + str(opt.resume) + '.pt')))

    # training data
    train_dataset = TrainDataset(opt.data_path, opt.gt_path, opt.imageSize, opt.batchSize * 3000)

    # train model
    train_model(netG, netD, train_dataset, optimizerG, schedulerG, optimizerD, schedulerD, opt.resume)


if __name__ == '__main__':
    main("BN")

