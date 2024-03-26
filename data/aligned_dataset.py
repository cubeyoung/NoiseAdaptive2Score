import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import imageio as io
import random
class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt,phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.phase = phase
        self.dir_A = os.path.join(opt.dataroot)  
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
    def patch(self, A):
        A = self.get_patch(
                    A,
                    patch_size=256,
                    scale=1,
                    multi=False,
                    input_large=False
                )
        #img = common.augment(img)
        return A
    def set_phi(self, iter):       
        min_log = np.log([20/255])
        self.phi_now = 55/255
        phi_s = min_log + np.random.rand(1) * (np.log([self.phi_now]) - min_log)
        phi_s = np.exp(phi_s) 
    
        return phi_s
    def generate_gamma(self,hr,zeta):
        zeta = zeta**2
        #print('Gamma scale {}'.format(1/zeta))
        lr = (hr)*np.random.gamma(1/zeta,zeta,hr.shape)
        return lr
    
    def generate_normal(self,hr,sigma):
        #print('Gaussian scale {}'.format(sigma*255))
        lr = hr + np.random.normal(size = hr.shape,scale = sigma)
        return lr        
    
    def generate_poi(self,hr,eta):
        eta = eta**2
        #print('Poisson scale {}'.format(eta))
        hr = np.clip(hr,0,1)
        lr = np.random.poisson(hr/eta) * eta 
        return lr
    
    def get_patch(self,*args, patch_size=96, scale=2, multi=False, input_large=False):
        ih, iw = args[0].shape[:2]

        if not input_large:
            p = scale if multi else 1
            tp = p * patch_size
            ip = tp // scale
        else:
            tp = patch_size
            ip = patch_size

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        if not input_large:
            tx, ty = scale * ix, scale * iy
        else:
            tx, ty = ix, iy

        ret = [
            args[0][iy:iy + ip, ix:ix + ip,:],
            *[a[ty:ty + tp, tx:tx + tp,:] for a in args[1:]]
        ]

        return ret  
    def augment(self,args, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
            return img

        return [_augment(a) for a in args] 
    
    def add_noise(self,args,phi, normal=True, poi=True):
        normal = normal and random.random() < 0.5
        poisson = poi and random.random() < 0.5
        gamma = poi and random.random() < 0.5
        def _add_noise(img):
            if normal:
                img = self.generate_normal(img,phi)
            if poisson: 
                img = self.generate_poi(img,phi)
            if gamma: 
                img = self.generate_gamma(img,phi)
            return img

        return [_add_noise(a) for a in args] 
    
    def np2Tensor(self,args):
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()

            return tensor

        return [_np2Tensor(a) for a in args]
    
    def _get_index(self, idx):
        if self.phase == 'train':
            return idx % len(self.A_paths)
        else:
            return idx    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        index = self._get_index(index)
        #phi = self.set_phi(index)
        A_path = self.A_paths[index]
        A = io.imread(A_path)/255        
        if self.phase == 'train' or self.phase == 'valid':
            A = self.patch(A)       
            A = self.augment(A)
        #B = self.add_noise(A,phi)
        A = self.np2Tensor(A)[0]
        #B = self.np2Tensor(B)[0]
        return {'A': A,'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.phase == 'train':
            return len(self.A_paths)*20
        elif self.phase == 'test':
            return len(self.A_paths)
        else:
            return len(self.A_paths)