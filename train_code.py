from __future__ import division
import os
import time
import glob
import datetime
import argparse
import shutil
import numpy as np
from crip.io import imwriteTiff, imwriteRaw, imreadRaw, listDirectory, imreadTiffs, fetchCTParam, readDicom, imreadTiff
from crip.postprocess import fovCrop, muToHU, huToMu
from crip.physics import Spectrum, Atten, forwardProjectWithSpectrum, computeMu, EnergyConversion
from crip.spec import deDecompRecon
import numpy as np
import torch
import torch.nn.functional as F
from gecatsim.pyfiles.GetMu import GetMu
from torch import nn
from collections import OrderedDict

from clear_unet import UnetL3
def changeBaseSVDTorch(mu1Low, mu1High, mu2Low, mu2High, sigmaLow, sigmaHigh):
    ''' Compute the change of basis matrix using proposed method (noise level normalized).
    '''
    v1 = np.array([mu1Low, mu1High])
    v2 = np.array([mu2Low, mu2High])
    M = np.array([v1, v2]).T
    A = np.linalg.inv(M) @ np.diag([sigmaLow, sigmaHigh])
    U, S, VT = np.linalg.svd(A)
    S = np.diag(S)
    D = S @ VT @ np.diag([1 / sigmaLow, 1 / sigmaHigh])
    D = np.linalg.inv(D)

    return D[:, 1], D[:, 0]

import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from arch_unet import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_channel', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--patchsize2', type=int, default=17)
parser.add_argument("--Lambda3", type=float, default=1.0)
parser.add_argument("--Lambda4", type=float, default=10.0)
parser.add_argument("--Lambda7", type=float, default=0.1)
parser.add_argument("--increase_ratio", type=float, default=2.0)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
Water = Atten.fromBuiltIn('Water')
Iodine = Atten.fromBuiltIn('I')
SpecLow = Spectrum.monochromatic(80)
SpecHigh = Spectrum.monochromatic(120)
MuWaterLow = computeMu(Water, SpecLow, EnergyConversion.EID)
MuWaterHigh = computeMu(Water, SpecHigh, EnergyConversion.EID)
MuIodineLow = computeMu(Iodine, SpecLow, EnergyConversion.EID)
MuIodineHigh = computeMu(Iodine, SpecHigh, EnergyConversion.EID)
v1 = [MuWaterLow, MuWaterHigh]
v2_iodine = [MuIodineLow, MuIodineHigh]

v1prim, v2prim = changeBaseSVDTorch(*v1, *v2_iodine, low_std, high_std)  # para: low_std high_std

def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)



def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


import numpy as np
import torch.utils.data as data
import torch
import cv2
import random
from pydicom import dcmread
import torch.nn.functional as F


def random_rotate_mirror(img_0, random_mode):
    if random_mode == 0:
        img = img_0
    elif random_mode == 1:
        img = img_0[::-1, ...]
    elif random_mode == 2:
        img = cv2.rotate(img_0, cv2.ROTATE_90_CLOCKWISE)
    elif random_mode == 3:
        img_90 = cv2.rotate(img_0, cv2.ROTATE_90_CLOCKWISE)
        img = img_90[:, ::-1, ...]
    elif random_mode == 4:
        img = cv2.rotate(img_0, cv2.ROTATE_180)
    elif random_mode == 5:
        img_180 = cv2.rotate(img_0, cv2.ROTATE_180)
        img = img_180[::-1, ...]
    elif random_mode == 6:
        img = cv2.rotate(img_0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif random_mode == 7:
        img_270 = cv2.rotate(img_0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = img_270[:, ::-1, ...]
    else:
        raise TypeError
    return img


def read_files(data_file):
    data_files = []
    fid = open(data_file, 'r')
    lines = fid.readlines()
    for l in lines:
        file_l = l.split()
        data_files.append(file_l)
    return data_files


def read_dicom(fpath):
    with open(fpath, 'rb') as infile:
        ds = dcmread(infile)
    data = ds.pixel_array
    return data


def random_rotate_mirror(img_0, random_mode):
    if random_mode == 0:
        img = img_0
    elif random_mode == 1:
        img = torch.flip(img_0, [1])
    elif random_mode == 2:
        img = torch.rot90(img_0, 1, [2, 1])
    elif random_mode == 3:
        img_90 = torch.rot90(img_0, 1, [2, 1])
        img = torch.flip(img_90, [2])
    elif random_mode == 4:
        img = torch.rot90(img_0, 2, [2, 1])
    elif random_mode == 5:
        img_180 = torch.rot90(img_0, 2, [1, 2])
        img = torch.flip(img_180, [1])
    elif random_mode == 6:
        img = torch.rot90(img_0, 1, [1, 2])
    elif random_mode == 7:
        img_270 = torch.rot90(img_0, 1, [1, 2])
        img = torch.flip(img_270, [2])
    else:
        raise TypeError
    return img


from crip.io import imwriteTiff


def read_files(data_file):
    data_files = []
    fid = open(data_file, 'r')
    lines = fid.readlines()
    for l in lines:
        file_l = l.split()
        data_files.append(file_l)
    return data_files


def read_raw(fpath):
    channel = int(os.path.getsize(fpath) / 4 / 512 / 512)
    data = np.fromfile(fpath, dtype=np.float32)
    data = data.reshape((channel, 512, 512))
    return data

def deDecompTorch(low, high, mu1Low, mu1High, mu2Low, mu2High):
    ''' DECT Two-material decomposition in PyTorch. [BHW, BHW] -> [B2HW].
    '''
    A = torch.tensor([
        [mu1Low, mu2Low],
        [mu1High, mu2High],
    ], dtype=torch.float32).cuda()
    M = torch.linalg.inv(A)

    def decompSingle(low, high):
        c1 = M[0, 0] * low + M[0, 1] * high
        c2 = M[1, 0] * low + M[1, 1] * high
        return c1, c2

    res = torch.zeros((low.shape[0], 2, low.shape[1], low.shape[2])).cuda()
    for i in range(low.shape[0]):
        c1, c2 = decompSingle(low[i], high[i])
        res[i, 0, ...] = c1
        res[i, 1, ...] = c2

    return res


def normalize(x, l=0, r=3000):
    '''Normalization for RED-CNN.'''
    return (x - l) / (r - l)


def normalize_min_max(x):
    min_ = x.min()
    max_ = x.max()
    return (x - min_) / (max_ - min_), min_, max_


def normalize_mean_var(x):
    min_ = x.mean()
    max_ = x.var()
    return (x - min_) / max_, min_, max_


def denormalize(x, l=0, r=3000):
    '''Denormalization for RED-CNN.'''
    return x * (r - l) + l


def pearsonR(x, y, batch_first=True, eps=1e-6):
    '''Pearson Correlation Coefficient in PyTorch
    '''
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True) + eps
    y_std = y.std(dim=dim, keepdim=True) + eps

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr


class LDCT(data.Dataset):

    def __init__(self, data_file, crop_size=256, crop_size2 = 13,
                 hu_range=None, mu=None):
        if hu_range is None:
            hu_range = [-1300, 2000]
        if mu is None:
            Water = Atten.fromBuiltIn('Water')
            SpecLow = Spectrum.monochromatic(80)
            SpecHigh = Spectrum.monochromatic(120)
            MuWaterLow = computeMu(Water, SpecLow, EnergyConversion.EID)
            MuWaterHigh = computeMu(Water, SpecHigh, EnergyConversion.EID)
            mu = [MuWaterLow, MuWaterHigh]
        self.mu = mu
        self.crop_size = crop_size
        self.crop_size2 = crop_size2
        self.data_files = read_files(data_file)
        self.range = hu_range

    def random_crop_img3(self, img_list):
        y = torch.randint(0, img_list[0].shape[1] - self.crop_size2 + 1, (1,))
        x = torch.randint(0, img_list[0].shape[2] - self.crop_size2 + 1, (1,))
        img_out = []
        for img in img_list:
            img_out.append(img[:, y: y + self.crop_size2, x: x + self.crop_size2])
        return img_out[0]

    def mu2hu(self, image):
        image[0:3] = (image[0:3] - self.mu[0]) / self.mu[0] * 1000.
        image[3:] = (image[3:] - self.mu[1]) / self.mu[1] * 1000.
        return image

    def __getitem__(self, index):
        path_ldct_1, path_ldct_2, label, _, _ = self.data_files[index]

        img_noise_1 = read_raw(path_ldct_1)
        img_noise_2 = read_raw(path_ldct_2)

        img_noise_1 = img_noise_1.astype(np.float32)
        img_noise_2 = img_noise_2.astype(np.float32)

        img_noise_1 = self.mu2hu(img_noise_1)
        img_noise_2 = self.mu2hu(img_noise_2)

        img_noise_1 = np.clip(img_noise_1, self.range[0], self.range[1])
        img_noise_2 = np.clip(img_noise_2, self.range[0], self.range[1])

        img_noise_1 = (img_noise_1 - self.range[0]) / (self.range[1] - self.range[0])
        img_noise_2 = (img_noise_2 - self.range[0]) / (self.range[1] - self.range[0])

        img_noise_1 = torch.from_numpy(img_noise_1).to(torch.float32)
        img_noise_2 = torch.from_numpy(img_noise_2).to(torch.float32)
        
        if self.crop_size2 is not None:
            img_noise_1_small = self.random_crop_img3([img_noise_1])

        return img_noise_1, img_noise_2, img_noise_1_small

    def __len__(self):
        return len(self.data_files)


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()




def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

def get_gaussian_kernel(kernel_size, sigma):
    n = (torch.arange(0, kernel_size) - (kernel_size - 1.0) / 2.0).unsqueeze(-1)
    var = 2 * (sigma ** 2).unsqueeze(-1) + 1e-8  # add constant for stability

    kernel_1d = torch.exp((-n ** 2) / var.t()).permute(1, 0)

    kernel_2d = torch.bmm(kernel_1d.unsqueeze(2), kernel_1d.unsqueeze(1))

    kernel_2d /= kernel_2d.sum(dim=(-1, -2)).view(-1, 1, 1)

    return kernel_2d

def HU2mu(x, mu_water):
    return (x/1000*mu_water+mu_water)

def denormalization2(x, energy=None):
    mu = x * 3300. - 1300.
    if energy is None:
        return HU2mu(mu[:,0:x.shape[1]//2,:,:], MuWaterLow), HU2mu(mu[:,x.shape[1]//2:,:,:], MuWaterHigh)
    else:
        if energy == 'L':
            return HU2mu(mu, MuWaterLow)
        else:
            return HU2mu(mu, MuWaterHigh)

def apply_gaussian_blur(tensor, kernel_size=5, sigma=0.8):
    gaussian_kernel = get_gaussian_kernel(kernel_size, torch.Tensor([sigma])).to(tensor.device).to(
        torch.float32).unsqueeze(0)
    channels = tensor.size(1)
    gaussian_kernel = gaussian_kernel.expand(channels, 1, kernel_size, kernel_size)  # 对每个通道使用同样的高斯核
    blurred_tensor = F.conv2d(tensor, gaussian_kernel, padding=kernel_size // 2, groups=channels)
    return blurred_tensor


TrainingDataset = LDCT('')
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

TestingDataset = LDCT('',crop_size=None)
TestingLoader = DataLoader(dataset=TestingDataset,
                            num_workers=2,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

# Network
from unet import UNet
network = UNet(in_chl=opt.n_channel,
               out_chl=opt.n_channel)
from redcnn import RED_CNN
from noise2sim_unet_ori import UNet2 as Noise2Sim_UNet
teacher = RED_CNN()
trainedDict: OrderedDict = torch.load('redcnn_torch_weight.pth')
teacher.load_state_dict(trainedDict, strict=False)

teacher2 = Noise2Sim_UNet(2,2,96,False,True,"leaky_relu",True)
checkpoint2 = torch.load('checkpoint_last_5e4.pth.tar',map_location='cpu')
state_dict = checkpoint2['state_dict']
for k in list(state_dict.keys()):
    if k.startswith('module.base_net.'):
        state_dict[k[len('module.base_net.'):]] = state_dict[k]
    del state_dict[k]
teacher2.load_state_dict(state_dict)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network = torch.nn.DataParallel(network).cuda()
teacher = torch.nn.DataParallel(teacher).cuda()
teacher2 = torch.nn.DataParallel(teacher2).cuda()
network = network.to(device)
teacher = teacher.to(device)
teacher2 = teacher2.to(device)

L1Loss = nn.L1Loss(reduction='mean')

# about training scheme
num_epoch = opt.n_epoch
ratio = num_epoch / 100
optimizer = optim.Adam(network.parameters(), lr=opt.lr)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=opt.gamma)

print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))
checkpoint(network, 0, "model")
print('init finish')
for epoch in range(1, opt.n_epoch + 1):
    cnt = 0

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()
    teacher.eval()
    teacher2.eval()
    for iteration, (noisy,clean, small_patch) in enumerate(TrainingLoader):
        st = time.time()
        clean = clean.cuda().to(torch.float32)
        noisy = noisy.cuda().to(torch.float32)
        small_patch = small_patch.cuda().to(torch.float32)
        optimizer.zero_grad()

        noisy_denoised = network(noisy)

        noisy_denoised_small_patch = network(small_patch)

        Lambda = epoch / opt.n_epoch * opt.increase_ratio

        loss1 = 0.0
        loss2 = 0.0
        with torch.no_grad():
            images_noise_ = noisy.reshape((-1,1,noisy.shape[-2],noisy.shape[-1]))
            outputs_teacher = teacher(images_noise_)
            outputs_teacher = outputs_teacher.reshape(noisy.shape)

            list1 = [0,3]
            list2 = [1,4]
            list3 = [2,5]
            outputs_teacher2 = torch.zeros_like(outputs_teacher)
            images_noise_ = noisy[:,list1,:,:]
            outputs_teacher2_part = teacher2(images_noise_)
            outputs_teacher2[:,list1,:,:] = outputs_teacher2_part

            images_noise_ = noisy[:,list2,:,:]
            outputs_teacher2_part = teacher2(images_noise_)
            outputs_teacher2[:,list2,:,:] = outputs_teacher2_part

            images_noise_ = noisy[:,list3,:,:]
            outputs_teacher2_part = teacher2(images_noise_)
            outputs_teacher2[:,list3,:,:] = outputs_teacher2_part

        
        if(opt.Lambda3 > 0):

            ref1,ref2 = denormalization2(outputs_teacher)
            ref1_2,ref2_2 = denormalization2(outputs_teacher2)
            e1,e2 = denormalization2(noisy_denoised) 
            refUT1 = deDecompTorch(ref1.squeeze().reshape((-1,ref1.shape[-2],ref1.shape[-1])), ref2.squeeze().reshape((-1,ref2.shape[-2],ref2.shape[-1])), *v1, *v2_iodine)
            refUT1_2 = deDecompTorch(ref1_2.squeeze().reshape((-1,ref1_2.shape[-2],ref1_2.shape[-1])), ref2_2.squeeze().reshape((-1,ref2_2.shape[-2],ref2_2.shape[-1])), *v1, *v2_iodine)
            refUT2 = deDecompTorch(e1.squeeze().reshape((-1,e1.shape[-2],e1.shape[-1])), e2.squeeze().reshape((-1,e2.shape[-2],e2.shape[-1])), *v1, *v2_iodine)
            refUT1[:,1,:,:] = refUT1[:,1,:,:] * 4.93 * 20
            refUT1_2[:,1,:,:] = refUT1_2[:,1,:,:] * 4.93 * 20
            refUT2[:,1,:,:] = refUT2[:,1,:,:] * 4.93 * 20

            e1_noisy, e2_noisy = denormalization2(noisy)
            refUT1_ca_teacher = deDecompTorch(ref1.squeeze().reshape((-1,ref1.shape[-2],ref1.shape[-1])), ref2.squeeze().reshape((-1,ref2.shape[-2],ref2.shape[-1])), *v1prim, *v2prim)[:, 0, ...].unsqueeze(1)
            refUT1_ca_teacher2 = deDecompTorch(ref1_2.squeeze().reshape((-1,ref1_2.shape[-2],ref1_2.shape[-1])), ref2_2.squeeze().reshape((-1,ref2_2.shape[-2],ref2_2.shape[-1])), *v1prim, *v2prim)[:, 0, ...].unsqueeze(1)
            refUT2_ca_noisy = deDecompTorch(e1_noisy.squeeze().reshape((-1,e1_noisy.shape[-2],e1_noisy.shape[-1])), e2_noisy.squeeze().reshape((-1,e2_noisy.shape[-2],e2_noisy.shape[-1])), *v1prim, *v2prim)[:, 0, ...].unsqueeze(1)    
            mask_teacher1 = (torch.abs(refUT1_ca_teacher-refUT2_ca_noisy)) > 0.02
            mask_teacher1 = 1 - mask_teacher1.to(torch.float32).detach()

            mask_teacher2 = (torch.abs(refUT1_ca_teacher2-refUT2_ca_noisy)) > 0.02
            mask_teacher2 = 1 - mask_teacher2.to(torch.float32).detach()

            loss3_2 = (L1Loss(refUT1.detach()*mask_teacher1,refUT2*mask_teacher1) + L1Loss(refUT1_2.detach()*mask_teacher2,refUT2*mask_teacher2) )* opt.Lambda3 * (1-(epoch / opt.n_epoch)) / 2.

            loss3_2_term = np.mean(loss3_2.item())
            loss3 = 0.0
            loss3_term = 0.0
        else:
            loss3 = 0.0
            loss3_term = 0.0
            loss3_2 = 0.0
            loss3_2_term = 0.0
        if(opt.Lambda4 > 0):
            ref1, ref2 = denormalization2(noisy)
            e1, e2 = denormalization2(noisy_denoised)
            refUT1_ca = deDecompTorch(ref1.squeeze().reshape((-1,ref1.shape[-2],ref1.shape[-1])), ref2.squeeze().reshape((-1,ref2.shape[-2],ref2.shape[-1])), *v1prim, *v2prim)[:, 0, ...].unsqueeze(1)
            refUT2_ca = deDecompTorch(e1.squeeze().reshape((-1,e1.shape[-2],e1.shape[-1])), e2.squeeze().reshape((-1,e2.shape[-2],e2.shape[-1])), *v1prim, *v2prim)[:, 0, ...].unsqueeze(1)    
            loss4 = L1Loss(refUT1_ca.detach(),refUT2_ca) * opt.Lambda4
            loss4_term = np.mean(loss4.item())
        else:
            loss4 = 0.0
            loss4_term = 0.0

        if(opt.Lambda7 > 0):
            e_low_small_patch, e_high_small_patch = denormalization2(noisy_denoised_small_patch)
            ref_low_small_patch, ref_high_small_patch = denormalization2(small_patch)
            e_low_small_patch = e_low_small_patch.reshape((-1,e_low_small_patch.shape[-2],e_low_small_patch.shape[-1]))
            e_high_small_patch = e_high_small_patch.reshape((-1,e_low_small_patch.shape[-2],e_low_small_patch.shape[-1]))
            ref_low_small_patch = ref_low_small_patch.reshape((-1,ref_low_small_patch.shape[-2],ref_low_small_patch.shape[-1]))
            ref_high_small_patch = ref_high_small_patch.reshape((-1,ref_high_small_patch.shape[-2],ref_high_small_patch.shape[-1]))
            ref_low_small_patch = ref_low_small_patch.unfold(1, 3, 1).unfold(2, 3, 1)
            ref_high_small_patch = ref_high_small_patch.unfold(1, 3, 1).unfold(2, 3, 1)
            e_low_small_patch = e_low_small_patch.unfold(1, 3, 1).unfold(2, 3, 1)
            e_high_small_patch = e_high_small_patch.unfold(1, 3, 1).unfold(2, 3, 1)
            B, nH, nW, H, W = e_low_small_patch.shape
            e_low_small_patch = e_low_small_patch.reshape((B * nH * nW, H * W))
            e_high_small_patch = e_high_small_patch.reshape((B * nH * nW, H * W))
            ref_high_small_patch = ref_high_small_patch.reshape((B * nH * nW, H, W))
            ref_low_small_patch = ref_low_small_patch.reshape((B * nH * nW, H, W))
            refUT1Patches_ = deDecompTorch(ref_low_small_patch, ref_high_small_patch, *v1prim,
                                        *v2prim)[:, 1, ...].reshape((B * nH * nW, H * W))
            refUT1Stds = torch.std(refUT1Patches_, dim=1) + 1e-7
            pearson = torch.abs(pearsonR(e_low_small_patch, e_high_small_patch)).squeeze()
            loss7 = torch.mean(1./(refUT1Stds*1000.) * (1 - pearson)) * opt.Lambda7 * (epoch / opt.n_epoch)
            loss7_term = np.mean(loss7.item())
        else:
            loss7 = 0.0
            loss7_term = 0.0

        loss_all = loss3 + loss4 + loss7

        loss_all.backward()
        optimizer.step()

        print(
                '{:04d} {:05d} Loss1={:.8f}, Lambda={}, Loss2={:.8f}, Loss3={:.8f}, Loss3_2={:.8f}, Loss4={:.8f}, Loss6={:.8f}, Loss7={:.8f}, Loss_Full={:.8f}, Time={:.4f}'
                .format(epoch, iteration, 0.0, Lambda,
                        0.0, loss3_term, loss3_2_term, loss4_term, loss6_term, loss7_term, np.mean(loss_all.item()),
                        time.time() - st))
        
    scheduler.step()

    
