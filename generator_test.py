# After finishing the joint training, testing the derain module BNet

import torch
import cv2
import os
import argparse

from matplotlib import pyplot as plt
from torch.autograd import Variable
from AAE import GNet
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./rain100L/test/small/norain", help='path to testing images')
parser.add_argument('--stage', type=int, default=3, help='the stage number of GNet')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--netG', default='./syn100lmodels/G_state_188.pt', help="path to trained GNet")
parser.add_argument('--save_path', default='./rainy_results/rain100L/', help='folder to rainy images')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
try:
    os.makedirs(opt.save_path)
except OSError:
    pass


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False


def normalize(data):
    return data / 255.


def main():
    # Build model
    print('Loading model ...\n')
    netG = GNet(opt.stage, 128, 32).cuda()
    netG.load_state_dict(torch.load(opt.netG))
    netG.eval()
    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            print(img_name)
            img_path = os.path.join(opt.data_path, img_name)
            # input testing rainy image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y)).cuda()
            with torch.no_grad():  #
                if opt.use_gpu:
                    torch.cuda.synchronize()
                start_time = time.time()

                rain_make, mu_z, logvar_z, _ = netG(y)
                # rain_make += y
                out = torch.clamp(rain_make, 0., 1.)

                if opt.use_gpu:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time
                print(img_name, ': ', dur_time)
                if opt.use_gpu:
                    save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())  # back to cpu
                else:
                    save_out = np.uint8(255 * out.data.numpy().squeeze())
            print(save_out.shape,mu_z.shape)
            for i in range(save_out.shape[0]):
                save = save_out[i].squeeze().transpose(1, 2, 0)
                b, g, r = cv2.split(save)
                save = cv2.merge([r, g, b])
                plt.imshow(save)
                plt.show()
                # cv2.imwrite(opt.save_path + img_name, save)
            count += 1
    print('Avg. time:', time_test / count)


if __name__ == "__main__":
    main()
