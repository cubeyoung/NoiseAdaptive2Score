#Train the network for Gaussian noise model
#python train.py --model Gaussian  --dataroot /your_path/ --dataroot_valid /your_path/  --name CBSD_ours_unet_gau --gpu_ids '0' --direction BtoA 

#Test the network for Gaussian noise model
#python test.py --model Gaussian  --dataset_mode test2  --noise_level 25 or 50 -dataroot /your_path/ --name CBSD_ours_unet_gau --model Gaussian --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/

#Train the network for Poisson noise model
#python train.py --model Poisson -  --dataroot /your_path/ --dataroot_valid /your_path/  --name CBSD_ours_unet_poi --gpu_ids '0' --direction BtoA 

#Test the network for Poisson noise model
#python test.py --model Poisson  --dataset_mode test2  --noise_level 001 or 005 --dataroot /your_path/  --name CBSD_ours_unet_poi --model Poisson --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/

#Train the network for Gamma noise model
#python train.py --model Gamma   --dataroot /your_path/ --dataroot_valid /your_path/  --name CBSD_ours_unet_gamma --gpu_ids '0' --direction BtoA 

#Test the network for Gamma noise model
#python test.py --model Gamma  --dataset_mode test2 --dataroot /your_path/ --name CBSD_ours_unet_gamma --model Gamma --direction BtoA  --gpu_ids '0' --epoch best --results_dir /your_results/ --noise_level g_100 or g_50


