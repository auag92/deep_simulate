import sys
import json
import numpy as np
from tqdm import tqdm
from toolz.curried import pipe, curry, compose

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from chnet.ch_losses import *
import chnet.ch_tools as ch_tools
import chnet.utilities as ch_utils
from chnet.models import get_model
import chnet.ch_generator as ch_gen
from chnet.ch_loader import CahnHillDataset


def get_criterion(criterion="mae", scale=100):
    if criterion == "mae":
        return mae_loss(scale=scale)
    elif criterion == "ssim":
        return ssim_loss(scale=scale)


def train(key="unet", 
          ngf=32,
          tanh=True,
          conv=True,
          mid=0.0, 
          dif=0.449, 
          dim_x=96, 
          dx=0.25, 
          dt=0.01, 
          gamma=0.2, 
          init_steps=1, 
          nstep=5, 
          n_samples_trn=1024, 
          n_datasets=10, 
          final_tstep=1001, 
          num_epochs=10, 
          learning_rate=1.0e-5,
          eta_min=1.0e-6,
          optimizer="sgd", 
          schedule=True,
          schedule_per_set=True,
          criterion="mae",
          scale=100,
          device="cuda", 
          save=True, 
          tag="script", 
          fldr="weights", 
          weight_file=None):
    
    device = torch.device("cuda:0") if device == "cuda" else torch.device("cpu")
    print(device)
    model = get_model(key=key, ngf=ngf, tanh=tanh, conv=conv, nstep=nstep, device=device)
    if len(weight_file) >0:
        print("loading saved weights")
        model.load_state_dict(torch.load(weight_file, map_location=device)["state"])
    
    delta_sim_steps=(final_tstep-init_steps)//nstep
    primes = ch_utils.get_primes(50000)[:n_datasets]
    print("no. of datasets: {}".format(len(primes)))
    fout = "{}/model_{}_size_{}_step_{}_init_{}_delta_{}_tstep_{}_tanh_{}_loss_{}_tag_{}.pt".format(fldr, key, ngf, nstep, init_steps, delta_sim_steps, num_epochs*len(primes), tanh, criterion, tag)  
    ffin = "{}/model_{}_size_{}_step_{}_init_{}_delta_{}_tstep_{}_tanh_{}_loss_{}_tag_{}_final.pt".format(fldr, key, ngf, nstep, init_steps, delta_sim_steps, num_epochs*len(primes), tanh, criterion, tag)  
    print("model saved at: {}".format(fout))

    print("Start Training")
    trn_losses = []
    criterion= get_criterion(criterion=criterion, scale=scale)
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(criterion)
    print(optimizer)
    
    if schedule:
        print("starting scheduler")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(primes)*n_samples_trn*num_epochs//8, eta_min=eta_min, last_epoch=-1)
    
    loss_prev = 100
    
    for num, prime in enumerate(primes):
        # Loss and optimizer
        torch.cuda.empty_cache()
        x_trn, y_trn = ch_gen.data_generator(nsamples=n_samples_trn, 
                                      dim_x=dim_x, 
                                      init_steps=init_steps, 
                                      delta_sim_steps = delta_sim_steps,
                                      dx=dx, 
                                      dt=dt,
                                      m_l=mid-dif, 
                                      m_r=mid+dif,
                                      n_step=nstep,
                                      gamma=gamma, 
                                      seed=2513*prime,
                                      device=device)


        trn_dataset = CahnHillDataset(x_trn, y_trn, 
                                      transform_x=lambda x: x[:,None], 
                                      transform_y=lambda x: x[:,None])

        trn_loader = DataLoader(trn_dataset, 
                                batch_size=8, 
                                shuffle=True, 
                                num_workers=4)

        print("Training Run: {}".format(num+1))

        total_step = len(trn_loader)

        for epoch in range(num_epochs):  
            for i, item_trn in enumerate(tqdm(trn_loader)):
                
                model.train()
                
                if "loop" in key:
                    if "solo" in key:
                        x = item_trn['x'].to(device)
                    else:
                        x = item_trn['x'][:,0].to(device)
                    y_tru = item_trn['y'].to(device)
                else:
                    x = item_trn['x'][:,0].to(device)
                    y_tru = item_trn['y'][:,-1] .to(device) 
  
                y_prd = model(x)# Forward pass
                # means_inp = x.mean(axis=(1,2,3))
                # means_out = y_prd.mean(axis=(1,2,3))
                loss = criterion(y_tru, y_prd)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                trn_losses.append(loss.item())
                scheduler.step()
            loss_curr = np.mean(trn_losses[-total_step:])
            print ('Epoch [{}/{}], Training Loss: {:.2f}, Learning Rate: {:.3e}'.format(epoch+1, num_epochs, loss_curr, scheduler.get_lr()[0]))
            obj = {}
            obj["state"] = model.state_dict()
            obj["losses"] = trn_losses
            if loss_curr < loss_prev:
                loss_prev = loss_curr
                if save:
                    torch.save(obj, fout)
                    print("model saved at set: {}, epoch: {}".format(num+1, epoch+1))
    torch.save(obj, ffin)        
    print("End Training")
    return obj

if __name__ == "__main__":

    if len(sys.argv) > 1:
        for argv in sys.argv[1:]:
            with open(argv, 'r') as f:
                arguments = json.load(f)
            print(arguments)
            model = train(**arguments)
            print(arguments)
    else:
        print("please supply input arguments file.")
