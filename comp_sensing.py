import numpy as np
import pickle as pkl
import os
import parser
import matplotlib.pyplot as plt

import torch
from torchvision import datasets

import utils
import cs_dip
import baselines

NEW_RECONS = False

args = parser.parse_args('configs.json')

NUM_MEASUREMENTS_LIST, BASIS_LIST = utils.convert_to_list(args)

dataloader = utils.get_data(args)   # get dataset of images over which to iterate
avg_losses = []     # mse_pixel_loss averaged over all images

for num_measurements in NUM_MEASUREMENTS_LIST:

    args.NUM_MEASUREMENTS = num_measurements
    A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
    mse_pixel_losses = []   # average square loss per pixel

    for _, (batch, _, im_path) in enumerate(dataloader):
        x = batch.view(1,-1).cpu().numpy() #for larger batch, change first arg of .view()
        y = np.dot(x,A)


        for basis in BASIS_LIST:

            args.BASIS = basis

            # if utils.recons_exists(args, im_path): # if reconstruction exists for a given config
            #     continue
            NEW_RECONS = True

            if basis == 'csdip':
                estimator = cs_dip.dip_estimator(args)
            elif basis == 'dct':
                estimator = baselines.lasso_dct_estimator(args)
            elif basis == 'wavelet':
                estimator = baselines.lasso_wavelet_estimator(args)
            else:
                raise NotImplementedError

            x_hat, y_loss_traj, img_loss_traj = estimator(A,x,y,args)
            mse_pixel_loss = utils.get_mse_pixel_loss(x, x_hat.reshape(1,-1))     # average square loss per pixel (for PR)
            mse_pixel_losses.append(mse_pixel_loss)

            utils.save_reconstruction(x_hat, args, im_path)
            names = ["Measure_loss", "Mse_img_loss_pixel"]
            utils.save_loss_plots([y_loss_traj, img_loss_traj], names, args, im_path)

    avg_losses.append(np.average(mse_pixel_losses))

# Notify user after reconstructions completed

if NEW_RECONS == False:
    print('Duplicate reconstruction configurations. No new data generated.')
else:
    print('Reconstructions generated!')

# Create plot of loss vs sensing ratio
plt.plot(NUM_MEASUREMENTS_LIST, avg_losses)
plt.xlabel('m/n (sensing ratio)')
plt.ylabel('mse (per pixel)')
plt.title('mse loss vs sensing ratio')
directory = os.getcwd()+'/reconstructions/'+args.DATASET+'/csdip/loss_plot/'
plt.savefig(directory + 'loss_vs_measure.png')

mn_ratios = NUM_MEASUREMENTS_LIST/np.product(args.IMG_SIZE*args.IMG_SIZE)
np.save(directory + 'mn_ratios', mn_ratios)
np.save(directory + 'avg_losses', avg_losses)