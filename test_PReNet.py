import argparse
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from PReNet import PReNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Model_Test")
parser.add_argument("--modelDir", type=str, default="./Models/net_epoch10.pth", help='path of model')
parser.add_argument("--rainDir", type=str, default='./out/test/rain_', help='path of rain')
parser.add_argument("--gtDir", type=str, default='./out/test/norain', help='path of ground truth')
parser.add_argument("--outDir", type=str, default='./out/derain_', help='path of derain results')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
opt = parser.parse_args()


def main(rainDir, gtDir, outDir):
    # Build model
    model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    model = model.cuda()
    model.load_state_dict(torch.load(opt.modelDir))
    model.eval()

    # process data

    psnr_test = 0
    ssim_test = 0
    for f in os.listdir(rainDir):
        # image
        rain = cv2.imread(os.path.join(rainDir, f)) / 255
        rain = np.transpose(rain, (2, 0, 1))
        rain = np.expand_dims(rain, axis=0)
        rain = torch.Tensor(rain).cuda()

        gt = cv2.imread(os.path.join(gtDir, f)) / 255

        out, _ = model(rain)
        out = out.data.cpu().numpy().astype(np.float32)
        out = np.transpose(out[0, :, :, :], (1, 2, 0))

        psnr = peak_signal_noise_ratio(out, gt, data_range=1)
        ssim = structural_similarity(out, gt, data_range=1, channel_axis=2)
        psnr_test += psnr
        ssim_test += ssim

        cv2.imwrite(os.path.join(outDir, f), out * 255)
        # print("%s PSNR %f SSIM %f" % (f, psnr, ssim))
    psnr_test /= len(os.listdir(rainDir))
    ssim_test /= len(os.listdir(rainDir))
    return ssim_test, psnr_test


if __name__ == "__main__":
    rainDir = "./out/test/rain_"
    gtDir = "./out/test/norain"
    outDir = './out/derain_'
    ssim_test1, psnr_test1 = main(rainDir, gtDir, outDir)
    rainDir = "./out/test/rain"
    gtDir = "./out/test/norain"
    outDir = './out/derain'
    ssim_test2, psnr_test2 = main(rainDir, gtDir, outDir)
    print('Derain result on generated rainy images based clear Rain100L:\n SSIM: %f PSNR:%f' % (ssim_test1, psnr_test1))
    print('Derain result on real Rain100L images:\n SSIM: %f PSNR:%f' % (ssim_test2, psnr_test2))
