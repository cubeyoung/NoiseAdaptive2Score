import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from skimage.measure import compare_psnr
import warnings
warnings.filterwarnings('ignore')
from util.util import calc_psnr
import numpy as np
import math
from .ema import ExponentialMovingAverage
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.optimize as optimize
from numpy.random import rand
import random
    
class GammaModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')

        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['f','sigma']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['lr','score','recon']
        if self.isTrain:
            self.model_names = ['f']
        else:  # during test time, only load G
            self.model_names = ['f']
            self.visual_names = ['lr','score','recon']
        self.netf = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_f = torch.optim.Adam(self.netf.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_f)
        self.variance1 = (opt.sigma/255)**2
        self.batch = opt.batch_size  
        self.sigma_min = 1e-3
        self.sigma_max = 0.1
        self.sigma_annealing = 2*11000
        self.target_model = opt.target_model
        self.acc= 0
        self.sigmas = np.exp(np.linspace(np.log(self.sigma_max), np.log(self.sigma_min),self.sigma_annealing))
        self.sigmas = torch.from_numpy(self.sigmas)
        self.ema = ExponentialMovingAverage(self.netf.parameters(), decay=0.999)
      
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.hr = input['A' if AtoB else 'A']#.to(self.device,dtype = torch.float32)
        self.phi_s = self.phi_s
        self.lr = (self.hr.numpy())*np.random.gamma(self.phi_s,1/self.phi_s,self.hr.shape)
        self.image_paths = input['A_paths' if AtoB else 'A_paths']        
        self.hr = self.hr.to(self.device,dtype = torch.float32)
        self.lr = torch.from_numpy(self.lr).to(self.device,dtype = torch.float32)
        
    def set_input_val(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.hr = input['A' if AtoB else 'A'].to(self.device,dtype = torch.float32)
        self.lr =input['B' if AtoB else 'B'].to(self.device,dtype = torch.float32)
        self.image_paths = input['A_paths' if AtoB else 'B_paths'] 
        
    def set_phi(self, iter):       
        min_log = np.log([40])
        self.phi_now = 120
        phi_s = min_log + np.random.rand(1) * (np.log([self.phi_now]) - min_log)
        self.phi_s = np.exp(phi_s)
        
    def set_sigma(self, iter):
        labels = torch.randint(0, len(self.sigmas), (self.lr.shape[0],))
        self.sigma = self.sigmas[labels].view(self.lr.shape[0], *([1] * len(self.lr.shape[1:]))).to(self.device,dtype = torch.float32)
        self.loss_sigma = self.sigma[0]
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.ema.store(self.netf.parameters())
        self.ema.copy_to(self.netf.parameters())             
        k = torch.from_numpy((self.phi_s)).to(self.device,dtype = torch.float32)
        self.score = self.netf(self.lr,0)[0]
        nom = k*self.lr
        denom = (k-1) - self.lr*self.score
        self.recon = nom/denom        
        self.ema.restore(self.netf.parameters())
    def foward_estimation(self,noise_model):
        def estimate(noise_model):
            if noise_model == "Gaussian":
                self.recon = self.forward_search_gau()
            elif noise_model == "Poisson":
                self.recon = self.forward_search_poi()
            else:
                self.recon = self.forward_search_gamma()
            return self.recon
        return estimate(noise_model)
    def forward_search_gau(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.ema.load_state_dict(self.loaded_state)
        self.ema.copy_to(self.netf.parameters())        
        self.noise = 1e-5 * torch.randn(self.lr.shape).to(self.device,dtype = torch.float32)
        self.score = self.netf(self.lr,0)[0]        
        self.score_2 = self.netf(self.lr+self.noise,0)[0]              
        self.noise_level = -self.noise/(self.score_2 - self.score)
        self.noise_level = torch.clamp(self.noise_level,0,1)
        self.noise_level = np.sqrt(torch.median(self.noise_level).cpu().detach().numpy())
        self.noise_level = self.noise_level*255
        self.recon = self.lr +(self.noise_level/255)**2 *(self.score)
        return self.recon
    def forward_search_poi(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.ema.load_state_dict(self.loaded_state)
        self.ema.copy_to(self.netf.parameters())        
        self.noise = 1e-5 * torch.randn(self.lr.shape).to(self.device,dtype = torch.float32)
        self.score = self.netf(self.lr,0)[0]        
        self.score_2 = self.netf(self.lr+self.noise,0)[0]              
        c = self.noise/(self.score_2 - self.score)
        self.noise_level = -self.lr + torch.sqrt((self.lr)**2 - 2*c)
        self.noise_level = torch.median(self.noise_level).cpu().detach().numpy()
        self.noise_level = np.around(self.noise_level,decimals= 2)
        self.recon = (self.lr +self.noise_level/2)*torch.exp(self.noise_level*self.score)
        return self.recon
    def forward_search_gamma(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.ema.load_state_dict(self.loaded_state)
        self.ema.copy_to(self.netf.parameters())        
        self.noise = 1e-6 * torch.randn(self.lr.shape).to(self.device,dtype = torch.float32)
        self.score = self.netf(self.lr,0)[0]        
        self.score_2 = self.netf(self.lr+self.noise,0)[0]              
        a = (self.score_2 - self.score)
        b = 1/(self.lr+self.noise) - 1/(self.lr)
        self.noise_level = b/(a+b)
        self.noise_level = (torch.median(self.noise_level).cpu().detach().numpy())
        self.noise_level = np.around(self.noise_level,decimals= 2)
        nom = self.lr
        denom = (1-self.noise_level)- self.noise_level*self.lr*self.score
        self.recon = nom/denom
        return self.recon
    
    def noise_model_estimation(self,score):
        epsilon = 1e-5
        self.n = torch.randn(self.lr.shape).to(self.device,dtype = torch.float32)
        self.noise = epsilon * self.n
        y_e = self.lr+self.noise
        score_e = self.netf(y_e,0)[0]
        w = 2*(y_e*score_e - self.lr*score).cpu().detach().numpy() 
        a = torch.log(y_e/self.lr).cpu().detach().numpy()       
        b = (2*self.lr*score).cpu().detach().numpy()        
        ww = w/(b+2.2)
        idx = (ww <= 1e-5) & (ww >= -1e-5)
        w = w[idx]
        b = b[idx]
        w = np.nanmean(w)
        b = np.nanmean(b)
        first = a*(b-2)
        second = 4*a*(- 2*a*b + w)
        sqrt = (first)**2 - second
        sqrt = np.sqrt(sqrt)    
        p1 = (-first + sqrt)/(2*a)
        p2 = (-first - sqrt)/(2*a)
        p1 = np.nanmean(p1)
        p2 = np.nanmean(p2)
        p = max(p1,p2)
        P = max(p,0)
        return p
 
    def forward_estimate(self):
        self.ema.load_state_dict(self.loaded_state)
        self.ema.copy_to(self.netf.parameters()) 
        self.score = self.netf(self.lr,0)[0]
        self.thetas = []
        self.noise_levels = []
        self.theta = self.noise_model_estimation(self.score)
        if (self.theta >= 0) & (self.theta <0.9) :
            self.noise_model = 'Gaussian'
        elif self.theta >= 1.9:
            self.noise_model = "Gamma"
        elif (self.theta >= 0.9) & (self.theta <1.9):
            self.noise_model = 'Poisson' 
        if self.target_model == self.noise_model:
            self.acc +=1            
        print("The estimated noise model of this image is : ", self.noise_model)
        self.recon = self.foward_estimation(self.noise_model)
        print("The estimated noise level parameter is : {}".format(self.noise_level))
        return self.recon
    
    def forward_psnr(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with torch.no_grad():
            self.ema.store(self.netf.parameters())
            self.ema.copy_to(self.netf.parameters())  
            self.recon = self.forward_estimate()
            self.recon = torch.clamp(self.recon.detach().cpu(), 0, 1)
            self.hr = self.hr.detach().cpu()
            self.ema.restore(self.netf.parameters())
            psnr = calc_psnr(self.recon,self.hr)                
        return  psnr
    
    def backward_f(self):
        """Calculate GAN and L1 loss for the generator"""            
        _,self.loss_f = self.netf(self.lr,self.sigma)     
        self.loss_f.backward()
        
    def optimize_parameters(self):        
        self.optimizer_f.zero_grad()        # set G's gradients to zero
        self.backward_f()                   # calculate graidents for G
        torch.nn.utils.clip_grad_norm_(self.netf.parameters(), 1)        
        self.optimizer_f.step()              # udpate G's weights              
        self.ema.update(self.netf.parameters())
        with torch.no_grad():
            self.forward()