# Langevin Cooling (L-Cool)

This code repository reproduces the results for the paper "Langevin Cooling for Domain Translation".

**L-Cool: [Paper](https://arxiv.org/abs/2008.13723)

This code has utilized majority of the code from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/) and [Tiramisu](https://github.com/bfortuner/pytorch_tiramisu). The extension consists of implementation of Langevin dynamics which was written by Vignesh Srinivasan.

### Requirements
    pytorch 
    python >= 3
    cv2
    imageio

To utilize Langevin Dynamics for CycleGAN, perform the following steps:

## 1. CycleGAN

### Dataset

```
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

### Pre-trained Model

-  Download the trained model from CycleGAN
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```

- Or train a model from scratch
```
python train.py --dataroot ./datasets/horse2zebra --name maps_cyclegan --model cycle_gan
```

## 2. Denoising Autoencoder (DAE)
### Train a Tiramisu Model 

```
python train_dae.py --dataroot ./datasets/horse2zebra --name horses_dae --model dae --display_id 0 --gaussian_noise 0.3  --netG tiramisu_67 --checkpoints_dir ./checkpoints/
```
The model checkpoint is stored in the directory `checkpoints/horses_dae/`. The noise added to the input of the DAE can be varied with `--gaussian_noise`.

## 3. L-Cool
### Perform L-Cool
```
python test_dae_langevin.py --dataroot ./datasets/horse2zebra --name horses_dae --model dae --display_id 0 --gaussian_noise 0.3  --netG tiramisu_67 --checkpoints_dir ./checkpoints/ --langevin_steps 100 --step_size 0.005 --temp 0.001 --save_gifs
```

### Hyperparameters for L-Cool
- `--langevin_steps` Number of steps  
- `--step_size` Step size
- `--temp` Temperature

The results can be found in the directory `results_dae_langevin`.

### GIF
Optionally, gifs can be saved by using `--save_gifs`. 



## Citations

When using this code for your research, please cite our paper
```
@article{srinivasan2020langevin,
  title={Langevin Cooling for Domain Translation},
  author={Srinivasan, Vignesh and M{\"u}ller, Klaus-Robert and Samek, Wojciech and Nakajima, Shinichi},
  journal={arXiv preprint arXiv:2008.13723},
  year={2020}
}
@inproceedings{srinivasan2020benign,
  title={Benign Examples: Imperceptible Changes Can Enhance Image Translation Performance.},
  author={Srinivasan, Vignesh and M{\"u}ller, Klaus-Robert and Samek, Wojciech and Nakajima, Shinichi}
  booktitle={AAAI},
  pages = {5842-5850},
  year={2020}     
}
```
