import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pdb 

class DAEModel(BaseModel):
    """
    This class implements the DAE model.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG tiramisu_67' ResNet generator,

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        A (source domain) r is the reconstruction function
        Generators: G_A: A -> r(A); 
        MSE loss:  ||(G_A(A) - A||^2 
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['cycle_A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'noisy_A', 'rec_A']
        #visual_names_B = ['real_B', 'fake_A', 'rec_B']

        self.gaussian_noise = opt.gaussian_noise
        self.visual_names = visual_names_A  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        self.netG_A = networks.define_T(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionCycle = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        
        self.noisy_A = self.gaussian(self.real_A, mean = 0, stddev = self.gaussian_noise)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def gaussian(self, ins, mean=0.0, stddev=0.25):
        if self.isTrain:
            #pdb.set_trace()
            noise = torch.empty(ins.shape).normal_(mean=mean,std=stddev)
            #noise = torch.randn_like(ins)
            noisy_ins = ins + noise.to(self.device)
            noisy_ins = torch.clamp(noisy_ins, -1,1)
            return noisy_ins
        else:
            return ins

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.rec_A = self.netG_A(self.noisy_A)  # G_A(A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
                        
        # Forward cycle loss || G_A(A) - A||^2
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) 
        # combined loss and calculate gradients
        self.loss_G = self.loss_cycle_A
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
