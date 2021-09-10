import os
# 3rd party
import torch
import numpy as np
import imageio
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
import h5py

# Project files.
import helpers

# input: img -> H x W x 3 
# output: mask -> H x W: T/F
#         foreground_pts -> n x 2

def get_foreground_points_white(img, sample_size=1024, threshold=0.01):
    H,W = img.shape[0], img.shape[1]
    
    img_color = img.sum(-1)
    mask = img_color > threshold
    foreground_pts = (np.where(mask)[0]/H, np.where(mask)[1]/W)
    foreground_pts_normalized = torch.from_numpy(np.array((list(zip(foreground_pts)))).squeeze(1).transpose()).float()
    colored_pts_normalized = img[mask]
    pts = np.concatenate((foreground_pts_normalized, colored_pts_normalized), -1)
    selected_idx = np.random.choice(pts.shape[0], sample_size)
    pts = pts[selected_idx]
    
    return pts

def get_foreground_points(img, sample_size=1024, threshold=2e-1):
    H,W = img.shape[0], img.shape[1]
    
    img_color = img.sum(-1)
    background_color = img_color[0][0]
    mask = (img_color > (background_color + threshold)) | (img_color < (background_color - threshold))
    foreground_pts = (np.where(mask)[0]/H, np.where(mask)[1]/W)
    foreground_pts_normalized = torch.from_numpy(np.array((list(zip(foreground_pts)))).squeeze(1).transpose()).float()
    colored_pts_normalized = img[mask]
    pts = np.concatenate((foreground_pts_normalized, colored_pts_normalized), -1)
    selected_idx = np.random.choice(pts.shape[0], sample_size)
    pts = pts[selected_idx]
    
    return pts

def apply_transform(shape, t=np.array([0.0,0.0]), s=1.0, r=0, norm_factor=255):

    out_mean = shape.mean(0)
    out = shape - out_mean

    cos_t   = np.cos(np.deg2rad(r))
    sin_t   = np.sin(np.deg2rad(r))
    
    # create rotation matrix using only pytorch functions
    r_mat = np.zeros((2,2))
    r_mat[0, 0] = cos_t
    r_mat[0, 1] = sin_t 
    r_mat[1, 0] = - sin_t
    r_mat[1, 1] = cos_t

    r_mat = r_mat.reshape(2, 2)

    out = out@r_mat

    out = out + out_mean
    out = out * s

    out = out + t

    return out


# general purpose dataloader
class TempFramePairs(torch.utils.data.Dataset):
    def __init__(self, data_sample=1, sample_size=1200, data_len=20):
        self.dataset_path = "/home/yichenl/source/layers/data/examples/example_{}/".format(data_sample)
        self.num_files = os.listdir(self.dataset_path)
        self.data_length = min(data_len, len(self.num_files))
        self.sample_size = sample_size
        self.paths = [self.dataset_path+"seq_{}.png".format(i) for i in range(self.data_length+1)]

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):

        seq = {}
        img1 = torch.from_numpy(np.asarray(imageio.imread(self.paths[idx]))[:,:,:3])
        img1 = (1.0 - img1/255.0).float()
        seq['img1'] = img1
        pts1 = get_foreground_points(img1, sample_size=self.sample_size)
        seq['pts1'] = pts1
        idx2 = min(idx + 1, len(self) - 1)
        img2 = torch.from_numpy(np.asarray(imageio.imread(self.paths[idx2]))[:,:,:3])
        img2 = (1.0 - img2/255.0).float()
        seq['img2'] = img2
        pts2 = get_foreground_points(img2, sample_size=self.sample_size)
        seq['pts2'] = pts2  


        return seq

# dataloader for making dummy examples
class TempFramePairsFixed(torch.utils.data.Dataset):
    def __init__(self, data_sample=1, sample_size=1200, data_len=20):
        self.dataset_path = "/home/yichenl/source/layers/data/examples/example_{}/".format(data_sample)
        self.num_files = os.listdir(self.dataset_path)
        self.data_length = min(data_len, len(self.num_files))
        self.sample_size = sample_size
        self.paths = [self.dataset_path+"seq_{}.png".format(i) for i in range(self.data_length)]
        self.patch = get_foreground_points(1.0 - np.asarray(imageio.imread(self.paths[0]))[:,:,:3]/255.0, sample_size=self.sample_size)
        self.patch[:,:2] = self.patch[:,:2] - self.patch[:,:2].min(0)

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, idx):

        seq = {}
        img1 = np.asarray(imageio.imread(self.paths[idx]))[:,:,:3]
        img1 = 1.0 - img1/255.0
        pts1 = apply_transform(self.patch[:,:2], r=2.0*idx)
        pts1 = np.concatenate([pts1, self.patch[:, 2:]], 1)
        seq['img1'] = torch.from_numpy(img1).float()
        seq['pts1'] = torch.from_numpy(pts1).float()
        idx2 = min(idx + 1, len(self) - 1)
        img2 = np.asarray(imageio.imread(self.paths[idx2]))[:,:,:3]
        img2 = 1.0 - img2/255.0
        translation2 = np.array([2.0*idx2, 2.0*idx2])
        pts2 = apply_transform(self.patch[:,:2], r=2.0*idx2) 
        pts2 = np.concatenate([pts2, self.patch[:, 2:]], 1)
        seq['img2'] = torch.from_numpy(img2).float()
        seq['pts2'] = torch.from_numpy(pts2).float()


        return seq


