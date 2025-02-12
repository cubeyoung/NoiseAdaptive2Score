# Noise Distribution Adaptive Self-Supervised Image Denoising using Tweedie Distribution and Score Matching [CVPR2022]

This repository is the official implementation of Noise Distribution Adaptive Self-Supervised Image Denoising using Tweedie Distribution and Score Matching. [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Noise_Distribution_Adaptive_Self-Supervised_Image_Denoising_Using_Tweedie_Distribution_and_CVPR_2022_paper.pdf) 

<img src="image/concept.png"  width="70%" height="70%" alt="Network"></img>

## Abstract
In this work, we provided a novel Bayesian framework for self-supervised image denoising without
clean data, which surpasses SURE, PURE, Noise2X, etc. Our novel innovation came from the
Tweedie’s formula, which provides explicit representation of denoise images through the score
function. By combining with the score-function estimation using AR-DAE, our Noise2Score can be
applied to image denoising problem from any exponential family noises. Furthermore, an identical
neural network training can be universally used regardless of the noise models, which leads to the
noise parameter estimation with minimal complexity. The links to SURE and existing Noise2X were
also explained, which clearly showed why our method is a better generalization.

<img src="image/intro.png"  width="700" height="370">

## Requirements

To install requirements:

```setup
conda env create -f requirements.yml
conda activate requirements
```

>📋  If you install anaconda package, it is possible to meet the prerequirements by running abobe code.

## Data
We generated synthetic noise images for each noise distribution. The trainset was set to DIVK2 and CBSD400. 
We provided the generation sourcecode "Datageneration.ipynb"


## Training

To train the model(s) in the paper for additive Gaussian noise, run this command:

```train
python train.py --model Gaussian  --dataroot /your_path/ --dataroot_valid /your_path/ --name CBSD_ours_unet_gau --gpu_ids '0' --direction BtoA 
```
To train the model(s) in the paper for Poisson noise, run this command:

```train
#python train.py --model Poisson -  --dataroot /your_path/ --dataroot_valid /your_path/  --name CBSD_ours_unet_poi --gpu_ids '0' --direction BtoA 
```

To train the model(s) in the paper for Gamma noise, run this command:

```train
#python train.py --model Gamma   --dataroot /your_path/ --dataroot_valid /your_path/  --name CBSD_ours_unet_gamma --gpu_ids '0' --direction BtoA 
```

>📋  Dataroot "your_path" depends on the your data path.

## Evaluation

To evaluate my model on test dataset for the Gaussian case, run:

```eval
#python test.py --model Gaussian  --dataset_mode test2  --noise_level 25 or 50 -dataroot /your_path/ --name CBSD_ours_unet_gau --model Gaussian --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/
```

To evaluate my model on test dataset for the Poisson case, run:

```eval
#python test.py --model Poisson  --dataset_mode test2  --noise_level 001 or 005 --dataroot /your_path/  --name CBSD_ours_unet_poi --model Poisson --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/
```

To evaluate my model on test dataset for the Gamma case, run:
```eval
python test.py --model Gamma  --dataset_mode test2 --dataroot /your_path/ --name CBSD_ours_unet_gamma --model Gamma --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/ --noise_level g_100 or g_50

```

>📋  Dataroot "your_path" depends on the your data path for test dataset such as CBSD68, Kodak. Change "--result_dir" to save results of image on your device. Specify your option to "noise level" 

## Pre-trained Models

You can download pretrained models [here](https://drive.google.com/drive/folders/15ap9SGq7WtXkRny9doGXfWaPdYGXppn3?usp=sharing) 

To brifely evaluate the proposed method given pretrained weight, we provided the Kodak Dataset for gaussian noisy and target pairs. 

Firrst, put in pretrained weights into checkpoints folder.

In case of Non-blind noise:

run:
```
python test.py --model Gaussian  --dataset_mode test2  --noise_level 25  -dataroot_valid /test_images/ --name CBSD_ours_unet_gau_blind --model Gaussian --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/
```

## Citation
If you find our work interesting, please consider citing
```
@inproceedings{kim2022noise,
  title={Noise distribution adaptive self-supervised image denoising using tweedie distribution and score matching},
  author={Kim, Kwanyoung and Kwon, Taesung and Ye, Jong Chul},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2008--2016},
  year={2022}
}
```


