import numpy as np
import parser
import torch
from torch.autograd import Variable
import baselines

import utils
from utils import DCGAN_XRAY, DCGAN_MNIST, DCGAN_CelebA, DCGAN_CelebA2


args = parser.parse_args('configs.json') # contains neural net hyperparameters

CUDA = torch.cuda.is_available()
dtype = utils.set_dtype(CUDA)

se = torch.nn.MSELoss(reduce=False).type(dtype)

BEGIN_CHECKPOINT = 50 # iteration at which to begin checking exit condition
EXIT_WINDOW = 25 # number of consecutive MSE values upon which we compare
NGF = 64
BATCH_SIZE = 1

meas_loss_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE))
reconstructions_ = np.zeros((args.NUM_RESTARTS, BATCH_SIZE, args.NUM_CHANNELS, \
                    args.IMG_SIZE, args.IMG_SIZE))

def dip_estimator(args):
    def estimator(A_val, x, y_batch_val, args):

        y = Variable(torch.Tensor(y_batch_val).type(dtype)) # cast measurements to GPU if possible
        temp_losses = []
        temp_img_losses = []

        for j in range(args.NUM_RESTARTS):
            if args.DATASET == 'xray':
                net = DCGAN_XRAY(args.Z_DIM, NGF, args.IMG_SIZE,\
                    args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
            elif args.DATASET == 'mnist':
                net = DCGAN_MNIST(args.Z_DIM, NGF, args.IMG_SIZE,\
                    args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
            elif args.DATASET == 'celeba':
                net = DCGAN_CelebA2(args.IMG_SIZE, args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
                # net = DCGAN_CelebA(args.Z_DIM, 128, args.IMG_SIZE,\
                #     args.NUM_CHANNELS, args.NUM_MEASUREMENTS)

            net.fc.requires_grad = False
            net.fc.weight.data = torch.Tensor(A_val.T) # set A to be fc layer

            allparams = [temp for temp in net.parameters()]
            allparams = allparams[:-1] # get rid of last item in list (fc layer)

            z = Variable(torch.zeros(BATCH_SIZE*args.Z_DIM).type(dtype).view(BATCH_SIZE,args.Z_DIM,1,1))
            z.data.normal_().type(dtype)
            z.requires_grad = False

            if CUDA:
                net.cuda()

            optim = torch.optim.RMSprop(allparams,lr=args.LR, momentum=args.MOM, weight_decay=args.WD)

            loss_temp = []
            img_loss_temp = []
            for i in range(args.NUM_ITER):

                optim.zero_grad()
                measurements = net.measurements(z, batch_size=BATCH_SIZE)

                if(args.MEASURE_TYPE == "phase_retrieval"):
                    measurements = measurements.abs()
                    y = y.abs()

                loss = torch.mean(torch.sum(se(measurements, y), dim=1))
                loss.backward()
                
                meas_loss = np.sum(se(measurements, y).data.cpu().numpy(),axis=1)[0]
                x_hat_temp = net(z).data.view(1,-1).cpu().numpy()
                img_loss = utils.get_mse_pixel_loss(x, x_hat_temp)
                img_loss_temp.append(img_loss)
                loss_temp.append(meas_loss) # save loss value of each iteration to array
                
                if (i >= BEGIN_CHECKPOINT): # if optimzn has converged, exit descent
                    should_exit, loss_min_restart = utils.exit_check(loss_temp[-EXIT_WINDOW:],i)
                    if should_exit == True:
                        meas_loss = loss_min_restart # get first loss value of exit window
                        break
                else:
                    should_exit = False

                optim.step()  


            temp_img_losses.append(img_loss_temp)   # save image loss trajectory for each restart
            temp_losses.append(loss_temp)   # save loss trajectory for each restart
            reconstructions_[j] = net(z).data.cpu().numpy() # get reconstructions        
            meas_loss_[j] = meas_loss # get last measurement loss for a given restart

        idx_best = np.argmin(meas_loss_,axis=0) # index of restart with lowest loss
        x_hat = reconstructions_[idx_best] # choose best reconstruction from all restarts
        y_loss_traj = temp_losses[idx_best[0]] # choose best loss trajectory from all restarts
        img_loss_traj = temp_img_losses[idx_best[0]] # choose best image loss trajectory from all restarts


        return x_hat, y_loss_traj, img_loss_traj

    return estimator
