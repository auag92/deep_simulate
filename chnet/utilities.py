import os
import numpy as np
import matplotlib.pyplot as plt
from toolz.curried import pipe, curry, compose
from mpl_toolkits.axes_grid1 import make_axes_locatable


def count_parameters(model):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def draw_im(im, title=None, vmin=-1, vmax=1):
    im = np.squeeze(im)
    plt.imshow(im, interpolation='nearest', cmap='seismic', vmin=vmin, vmax=vmax)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
    
@curry
def return_slice(x_data, cutoff):
    if cutoff is not None:
        return pipe(x_data,
                    lambda x_data: np.asarray(x_data.shape).astype(int) // 2,
                    lambda new_shape: [slice(new_shape[idim]-cutoff,
                                             new_shape[idim]+cutoff+1)
                                       for idim in range(x_data.ndim)],
                    lambda slices: x_data[slices])
    else:
        return x_data
    

def draw(X, title='', sample_id = 0):
    if X.ndim == 3:
        try:
            im = plt.imshow(X[sample_id][:][:], extent=[0, 1, 0, 1])
            plt.colorbar()
            plt.title(title)
            plt.show(im)
        except IndexError:
            print("Requested Sample # %d exceeds data size" % (sample_id+1))
    else:
        im = plt.imshow(X[:][:], extent=[0, 1, 0, 1])
        plt.colorbar()
        plt.title(title)
        plt.show(im)


def colorbar(mappable):
    """
    https://joseph-long.com/writing/colorbars/
    """
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

        
def draw_by_side(args, title='', sub_titles=None, scale=6, vmin=-1.0, vmax=1.0):
    fig, axs = plt.subplots(nrows=1, ncols=len(args), figsize=(scale*1.6180,scale))
    fig.suptitle(title, fontsize=20)
    
    
    for ix, arg in enumerate(args):
        if sub_titles:
            axs[ix].set_title(sub_titles[ix])
        im1 = axs[ix].imshow(arg, interpolation='nearest', cmap='seismic', vmin=vmin, vmax=vmax)
        colorbar(im1)
        
    for ix, ax in enumerate(axs[1:]):
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

    
def get_primes(n):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
    sieve[0] = False
    for i in range(int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)//3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]
