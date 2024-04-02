import torch
import numpy as np
import chnet.cahn_hill as ch
from toolz.curried import pipe, curry, compose


@curry
def init_norm(nsamples, dim_x, dim_y, seed=354875, m_l=-0.15, m_r=0.15):
    np.random.seed(seed)
    means  = np.random.uniform(m_l, m_r, size=nsamples)
    np.random.seed(seed)
    scales  = np.random.uniform(0.01, 0.02, size=nsamples)
    
    x_data = [np.random.normal(loc=m, scale=s, size = (1, dim_x, dim_y)) for m,s in zip(means, scales)]
    x_data = np.concatenate(x_data, axis=0)
    
    np.clip(x_data, -0.998, 0.998, out=x_data)
    
    return x_data


@curry
def data_generator(nsamples=2, 
                   dim_x=64, 
                   init_steps=1, 
                   delta_sim_steps=100,
                   dx=0.25, 
                   dt=0.01,
                   gamma=1.0, 
                   seed=None,
                   n_step=4,
                   m_l=-0.15, 
                   m_r=0.15,
                   device=torch.device("cpu")):
    
    init_data = init_norm(nsamples, dim_x, dim_x, seed=seed, m_l=m_l, m_r=m_r)   


    
    x_data = ch.ch_run_torch(init_data, 
                             dt=dt, gamma=gamma, 
                             dx=dx, sim_step=init_steps, 
                             device=device)    
    
    if n_step > 1:
        
        x_list = []
        y_list = []
        
        for _ in range(n_step):
            x_list.append(x_data[None])
            x_data = ch.ch_run_torch(x_data, 
                                     dt=dt, gamma=gamma, 
                                     dx=dx, sim_step=delta_sim_steps, 
                                     device=device)
            y_list.append(x_data[None])

        x_data = np.moveaxis(np.concatenate(x_list, axis=0), 0, 1)
        y_data = np.moveaxis(np.concatenate(y_list, axis=0), 0, 1)
        
    else:
        y_data = ch.ch_run_torch(x_data, 
                         dt=dt, gamma=gamma, 
                         dx=dx, sim_step=delta_sim_steps, 
                         device=device)

    return x_data, y_data