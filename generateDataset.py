
from random import randint
import torch
import cv2
import os
import argparse
from matplotlib import pyplot as plt
from torch.autograd import Variable
import time
import numpy as np
from Generator import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--rain_path", type=str, default="./rain100L/test/small/rain", help='path to rain images')
parser.add_argument("--norain_path", type=str, default="./rain100L/test/small/norain", help='path to norain images')
parser.add_argument("--out_path", type=str, default="./out/test", help='path to output images')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--netG', default='./Models/G_state_100.pt', help="path to trained GNet")
parser.add_argument('--save_path', default='./rainy_results/rain100L/', help='folder to rainy images')
parser.add_argument('--nc', type=int, default=3, help='Number of image channels')
parser.add_argument('--nz', type=int, default=128, help='size of noise z')
parser.add_argument('--nef', type=int, default=32, help='channel for Generator')
parser.add_argument('--num', type=int, default=100, help='size of generated dataset')

opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
try:
    os.makedirs(opt.save_path)
except OSError:
    pass


def transform(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img


def main():
    cropSize = [64, 64]
    # Build model
    print('Loading model ...\n')
    netG = Generator(opt.nc, opt.nz, opt.nef, "BN").cuda()
    netG.load_state_dict(torch.load(opt.netG))
    netG.eval()
    count = 0
    while count < opt.num:
        for img_name in os.listdir(opt.rain_path):
            if count == opt.num:
                break
            rain = cv2.imread(os.path.join(opt.rain_path, img_name)) / 255
            norain = cv2.imread(os.path.join(opt.norain_path, img_name)) / 255
            print(count, img_name)

            h = norain.shape[0]
            w = norain.shape[1]
            nh = randint(0, h - cropSize[0])
            nw = randint(0, w - cropSize[1])
            rain = rain[nh:nh + cropSize[0], nw:nw + cropSize[1], :]
            norain = norain[nh:nh + cropSize[0], nw:nw + cropSize[1], :]

            with torch.no_grad():
                if opt.use_gpu:
                    torch.cuda.synchronize()

                noise = torch.randn(1, opt.nz).cuda()

                mask = netG(noise)
                mask = np.transpose(mask.data.cpu().numpy().astype(np.float32)[0], (1, 2, 0))

                rain_ = norain + mask
                rain_ = np.clip(rain_, 0., 1.)

                if opt.use_gpu:
                    torch.cuda.synchronize()

            cv2.imwrite(os.path.join(opt.out_path, 'rain/' + str(count)+'.png'), rain*255)
            cv2.imwrite(os.path.join(opt.out_path, 'norain/' + str(count)+'.png'), norain*255)
            cv2.imwrite(os.path.join(opt.out_path, 'rain_/' + str(count)+'.png'), rain_*255)
            count += 1


if __name__ == "__main__":
    main()
