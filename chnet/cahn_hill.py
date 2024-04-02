import torch
import warnings
import numpy as np
import dask.array as da
from toolz.curried import pipe, curry, compose, memoize

try:
    import pyfftw
    np.fftpack = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
except ImportError:
    print("you can install PyFFTW for speed-up as - ")
    print("conda install -c conda-forge pyfftw")
    pass


fft = curry(np.fft.fft)
ifft = curry(np.fft.ifft)
fftn = curry(np.fft.fftn)
ifftn = curry(np.fft.ifftn)
fftshift = curry(np.fft.fftshift)
ifftshift = curry(np.fft.ifftshift)
torch_rfft = curry(torch.fft.rfft)
torch_irfft = curry(torch.fft.irfft)

def conjugate(x):
    y = torch.empty_like(x)
    y[..., 1] = x[..., 1] * -1
    y[..., 0] = x[... , 0]
    return y

@curry
def mult(x1, x2):
    y = torch.empty_like(x1)
    y[..., 0] = x1[..., 0]*x2[..., 0] - x1[..., 1]*x2[..., 1]
    y[..., 1] = x1[..., 0]*x2[..., 1] + x1[..., 1]*x2[..., 0]
    return y

@curry
def imfilter_real(f_data1, f_data2):
    """
    For convolving f_data1 with f_data2 using PyTorch
    """
    ndim = 2
    f_data1 = torch.from_numpy(f_data1).double()
    f_data2 = torch.from_numpy(f_data2).double()
    rfft = torch_rfft(signal_ndim=ndim)
    irfft = torch_irfft(signal_ndim=ndim)
    return pipe(f_data1,
                rfft,
                lambda x: mult(x, conjugate(rfft(f_data2))),
                irfft,
                fftshift)

@curry
def imfilter(x_data1, x_data2):
    """
    For convolving f_data1 with f_data2 using PyTorch
    """
    ndim = 2
    f_data1 = np.zeros(list(x_data1.shape)+[2,])
    f_data2 = np.zeros(list(x_data2.shape)+[2,])
    f_data1[...,0] = x_data1
    f_data2[...,0] = x_data2
    f_data1 = torch.from_numpy(f_data1).double()
    f_data2 = torch.from_numpy(f_data2).double()
    fft = curry(torch.fft.fft)(signal_ndim=ndim)
    ifft = curry(torch.fft.ifft)(signal_ndim=ndim)
    return pipe(f_data1,
                fft,
                lambda x: mult(x, conjugate(fft(f_data2))),
                ifft,
                lambda x: x[...,0],
                fftshift)

def ch_run_torch(x_data, gamma = 1., dx = 0.25, dt = 0.001, sim_step = 1, device = torch.device("cuda")):
    
    X = torch.tensor(x_data).to(device)
    N = X.shape[1]

    if not np.all(np.array(X.shape[1:]) == N):
        raise RuntimeError("X must represent a square domain")

    L = dx * N
    k = np.arange(N)

    if N % 2 == 0:
        N1 = N // 2
        N2 = N1
    else:
        N1 = (N - 1) // 2
        N2 = N1 + 1

    k[N2:] = (k - N1)[:N1]
    k = k * 2 * np.pi / L

    i_ = np.indices(X.shape[1:])
    ksq = np.sum(k[i_] ** 2, axis=0)[None]

    axes = (1, 2)

    FX = torch.fft.fftn(X, dim=axes)
    FX3 = torch.fft.fftn(X ** 3, dim=axes)

    a1 = 3.
    a2 = 0.

    explicit = ksq * (a1 - gamma * a2 * ksq)
    implicit = ksq * ((1 - a1) - gamma * (1 - a2) * ksq)


    Fy = (FX * (1 + dt * explicit) - ksq * dt * FX3) / (1 - dt * implicit)
    response = ifftn(Fy, axes=axes).real

    return response
        
    del FX, FX3, Fy, ksq, x_data
    torch.cuda.empty_cache()
    return response[...,0].cpu().numpy()


def ch_run(X, gamma = 1., dx = 0.25, dt = 0.001):
    N = X.shape[1]

    if not np.all(np.array(X.shape[1:]) == N):
        raise RuntimeError("X must represent a square domain")

    L = dx * N
    k = np.arange(N)

    if N % 2 == 0:
        N1 = N // 2
        N2 = N1
    else:
        N1 = (N - 1) // 2
        N2 = N1 + 1

    k[N2:] = (k - N1)[:N1]
    k = k * 2 * np.pi / L

    i_ = np.indices(X.shape[1:])
    ksq = np.sum(k[i_] ** 2, axis=0)[None]

    axes = np.arange(len(X.shape) - 1) + 1

    FX = fftn(X, axes=axes)
    FX3 = fftn(X ** 3, axes=axes)

    a1 = 3.
    a2 = 0.

    explicit = ksq * (a1 - gamma * a2 * ksq)
    implicit = ksq * ((1 - a1) - gamma * (1 - a2) * ksq)


    Fy = (FX * (1 + dt * explicit) - ksq * dt * FX3) / (1 - dt * implicit)
    response = ifftn(Fy, axes=axes).real

    return response

def _k_space(size):
    size1 = lambda: (size // 2) if (size % 2 == 0) else (size - 1) // 2
    size2 = lambda: size1() if (size % 2 == 0) else size1() + 1
    return np.concatenate(
        (np.arange(size)[: size2()], (np.arange(size) - size1())[: size1()]))

@memoize
def _calc_ksq_(shape):
    indices = lambda: np.indices(shape)
    return np.sum(_k_space(shape[0])[indices()] ** 2, axis=0)[None]


def _calc_ksq(x_data, delta_x):
    return _calc_ksq_(x_data.shape[1:]) * (2 * np.pi / (delta_x * x_data.shape[1])) ** 2


def _axes(x_data):
    return np.arange(len(x_data.shape) - 1) + 1


def _explicit(gamma, ksq, param_a1=3., param_a2=0.):
    return param_a1 - gamma * param_a2 * ksq


def _f_response(x_data, delta_t, gamma, ksq):
    fx_data = lambda: fftn(x_data, axes=_axes(x_data))
    fx3_data = lambda: fftn(x_data ** 3, axes=_axes(x_data))
    implicit = lambda: (1 - gamma * ksq) - _explicit(gamma, ksq)
    delta_t_ksq = lambda: delta_t * ksq
    numerator = (
        lambda: fx_data() * (1 + delta_t_ksq() * _explicit(gamma, ksq))
        - delta_t_ksq() * fx3_data()
    )
    return numerator() / (1 - delta_t_ksq() * implicit())


@curry
def solve(x_data, delta_x=0.25, delta_t=0.001, gamma=1.):
    return ifftn(
        _f_response(_check(x_data), delta_t, gamma, _calc_ksq(x_data, delta_x)),
        axes=_axes(x_data),
    ).real


def _check(x_data):
    """Ensure that domain is square.
    Args:
      x_data: the initial microstuctures
    Returns:
      the initial microstructures
    Raises:
      RuntimeError if microstructures are not square
    >>> _check(np.array([[[1, 2, 3], [4, 5, 6]]]))
    Traceback (most recent call last):
    ...
    RuntimeError: X must represent a square domain
    """
    if not np.all(np.array(x_data.shape[1:]) == x_data.shape[1]):
        raise RuntimeError("X must represent a square domain")

    return x_data
