import os
import argparse
import glob
import numpy as np
import torch
from Generator import Generator
import matplotlib.pyplot as plt
import random

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--nc', type=int, default=3, help='size of the RGB image')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--nef', type=int, default=32)
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
# parser.add_argument('--netG', default='./syn100lmodels/G_state_200.pt', help="path to netG for z--rain display")
# parser.add_argument('--save_fake', default='./disentanglement_results/rain100L/', help='folder to fake rain streaks')
parser.add_argument('--netG', default='./model/syn100lmodels_BN/G_state_100.pt',
                    help="path to netG for z--rain display")
parser.add_argument('--save_fake', default='./disentanglement_results_BN/rain100L/',
                    help='folder to fake rain streaks')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
try:
    os.makedirs(opt.save_fake)
except OSError:
    pass


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False


def main():
    # Build model
    print('Loading model ...\n')
    netG = Generator(opt.nc, opt.nz, opt.nef,"BN").cuda()
    netG.load_state_dict(torch.load(opt.netG))
    interpolation = torch.arange(-3, 3, 1)
    n = len(interpolation)
    img_size = 64
    figure_big = np.zeros((img_size * opt.nz, img_size * n, 3))
    multi_seed = np.arange(0, 1000, 100)
    # for seed in multi_seed:
    #     print(seed)
    #     random.seed(seed)
    #     torch.manual_seed(seed)
    #     random_z = torch.normal(0, 2, size=(1, opt.nz))
    #     random_z = random_z.cuda()
    #     with torch.no_grad():  #
    #         out = netG.sample(random_z)
    #         out = torch.clamp(out, 0., 1.)
    #     if opt.use_gpu:
    #         save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())  # back to cpu
    #     else:
    #         save_out = np.uint8(255 * out.data.numpy().squeeze())
    #     save_out = save_out.transpose(1, 2, 0)
    #     plt.imshow(save_out/255)
    #     plt.show()

    for seed in multi_seed:
        random.seed(seed)
        torch.manual_seed(seed)
        random_z = torch.normal(0, 1, size=(1, opt.nz))
        ori_z = random_z.clone()
        out_seed = opt.save_fake + './seed' + str(seed)  # save the disentanglement results with different seed
        try:
            os.makedirs(out_seed)
        except OSError:
            pass
        for d in range(opt.nz):
            count = 0
            figure = np.ones((img_size, img_size * n + 3 * (n - 1), 3)) * 255
            for val in interpolation:
                random_z[:, d] = val
                random_z = random_z.cuda()
                with torch.no_grad():  #
                    out = netG(random_z)
                    out = torch.clamp(out, 0., 1.)
                if opt.use_gpu:
                    save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())  # back to cpu
                else:
                    save_out = np.uint8(255 * out.data.numpy().squeeze())
                save_out = save_out.transpose(1, 2, 0)
                # print(save_out[:1, :1, :])
                star = count * img_size + 3 * (count)
                end = (count + 1) * img_size + 3 * (count)
                figure[0:img_size, star:end, :] = save_out
                figure_big[d * img_size:(d + 1) * img_size, count * img_size: (count + 1) * img_size, :] = save_out
                count += 1
            random_z = ori_z
            plt.imsave(out_seed + '/' + str(d) + '.png', figure / 255)
        plt.imsave(out_seed + '/' + 'fake.jpg', figure_big / 255)


if __name__ == "__main__":
    main()
