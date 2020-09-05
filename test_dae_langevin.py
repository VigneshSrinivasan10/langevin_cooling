"""
python test_dae_langevin.py --dataroot ./datasets/horse2zebra --name horses_dae --model dae --display_id 0 --gaussian_noise 0.3  --netG tiramisu_67 --checkpoints_dir ./checkpoints/ --langevin_steps 100 --step_size 0.005 --temp 0.001 
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_gif
from util import html
import pdb 
import ntpath
from PIL import Image
import torch
import numpy as np
import shutil
import time

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.results_dir = 'results_dae_langevin'
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.name = opt.name + '/' + str(opt.gaussian_noise)

    ### SET UP DAE
    dae_model = create_model(opt)      # create a model given opt.model and other options
    dae_model.setup(opt)               # regular setup: load and print networks; create schedulers

    dae_model.name = dae_model.opt.netG
    
    ### SET UP CYCLEGAN
    opt.netG = 'resnet_9blocks'
    opt.model = 'cycle_gan'

    # REUSE NAME FOR CYCLEGAN
    if 'horse' in opt.name:
        opt.name = 'horse2zebra_pretrained'
    elif 'zebra' in opt.name:
        opt.name = 'zebra2horse_pretrained'
    elif 'summer' in opt.name:
        opt.name = 'summer2winter_yosemite_pretrained'
    elif 'winter' in opt.name:
        opt.name = 'winter2summer_yosemite_pretrained'
    elif 'apple' in opt.name:
        opt.name = 'apple2orange_pretrained'
    elif 'orange' in opt.name:
        opt.name = 'orange2apple_pretrained'
    elif 'map' in opt.name:
        opt.name = 'sat2map_pretrained'
    else:
        NotImplementedError(' [%s] is not yet trained', opt.name)

    opt.checkpoints_dir = '../cyclegan/pytorch-CycleGAN-and-pix2pix/checkpoints'
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    opt.noise_std = (opt.temp * 2 * opt.step_size)**0.5
    opt.noise_std = float("{0:.4f}".format(opt.noise_std))
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, dae_model.name, str(opt.gaussian_noise),'temp{}_noisestd{}_stepsize{}'.format(opt.temp, opt.noise_std, opt.step_size), '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    sf_file_path = webpage.img_dir.replace('/images', '/files/')
    if os.path.exists(sf_file_path):
        shutil.rmtree(sf_file_path)
        
    os.makedirs(sf_file_path)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    direc = 'A'
    opp_direc = 'B'
    a2b = True if opt.direction == 'AtoB' else False
        
    for i, data in enumerate(dataset):
        all_visuals = {}
        gif_visuals = []
        
        #only_B = []
        for j in range(opt.langevin_steps):

            if j == 0:
                lang_data = data
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference

                vis = model.get_current_visuals()  # get image results
                visuals = torch.cat([vis['real_A'], vis['fake_B']] , -2)
                visuals = torch.cat([visuals, visuals] , -1)
                first_visuals = torch.cat([vis['real_A'], vis['fake_B']] , -2)
                input_image = vis['real_A']
                cyclegan = vis['fake_B']
                inp_cyclegan = torch.cat([input_image, cyclegan] , -2)
                gif_visuals += [visuals]
                
            dae_model.set_input(lang_data)  # unpack data from data loader
            dae_model.test()           # run inference
            dae_visuals = dae_model.get_current_visuals()  # get image results
            
            score_function = (dae_visuals['rec_A']  - dae_visuals['real_A']) / (opt.gaussian_noise**2)
            noise = torch.randn_like(score_function) * opt.noise_std
            lang_data['A' if a2b else 'B'] = lang_data['A' if a2b else 'B'].cuda() + opt.step_size * score_function.cuda() + noise.cuda()
            lang_data['A' if a2b else 'B'] = torch.clamp(lang_data['A' if a2b else 'B'], -1, 1)
            
            model.set_input(lang_data)  # unpack data from data loader
            model.test()           # run inference

            if j % 1 == 0:  # save images to an HTML file
                vis = model.get_current_visuals()  # get image results
                #visuals = torch.cat([vis['real_A'], cyclegan,  vis['fake_B']] , -1)
                visuals = torch.cat([vis['real_A'], vis['fake_B']] , -2)
                visuals = torch.cat([inp_cyclegan, visuals] , -1)
                gif_visuals += [visuals]

            img_path = model.get_image_paths()     # get image paths
            
            if j == opt.langevin_steps-1:  # save images to an HTML file
                #webpage.img_dir = webpage.img_dir.replace(opt.name,opt.name + '/' + str(opt.gaussian_noise))
                vis = model.get_current_visuals()  # get image results
                visuals = torch.cat([vis['real_A'], vis['fake_B']] , -2)
                visuals = torch.cat([inp_cyclegan, visuals] , -1)
                
                all_visuals['final'] = visuals
                gif_visuals += [visuals]
                print('processing (%04d)-th image and Langevin step: (%04d)... %s' % (i, j, img_path))
                save_images(webpage, all_visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
                if opt.save_gifs:
                    save_gif(webpage, gif_visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
