import numpy as np
from toolz.curried import pipe
from toolz.curried import curry


def initField_uniform(seed=101, shape=(1, 101, 101), scale=1e-1):
    """
    Generate a random field with uniform distribition around between -1 and 1
    """
    np.random.seed(seed)
    return (2 * np.random.random(shape) - 1)*scale


@curry
def sum_over_field(x_data, axis_ = (1,2)):
    """
    Sum over all points in a spatially discretized field.
    """
    return np.sum(x_data, axis=axis_)


def initField_normal(seed=101, shape=(1, 101, 101), scale=1e-1):
    """
    Generate a random field with uniform distribition around between -1 and 1.
    """
    np.random.seed(seed)
    return (np.random.normal(loc=0.0, scale=scale, size=shape))


def norm(x_data1, x_data2):
    """
    Returns the L2 norm for a 2D vector field when provided with the
    components at each point as two spatially discretized grids
    """
    return (x_data1**2 + x_data2**2)


def double_well(x_data):
    """
    Computes potential at each point in the concentration field using the double well formulation.
    """
    return (x_data**2 - 1)**2


@curry
def potential_field(x_data, delta_x, gamma):
    """
    return potential at each point in the concentration field
    """
    return 0.25 * double_well(x_data) + 0.5 * gamma * norm(*np.gradient(x_data, delta_x, axis = (1,2)))


@curry
def freeEnergy(x_data, delta_x, gamma):
    """
    Function to compute free energy at each point in spatially dicretized input
    concentration field.
    """
    return pipe(x_data,
               potential_field(delta_x=delta_x, gamma=gamma),
               sum_over_field(axis_=(1,2)))



def iterator(data, t_steps, solver, func, steps = None):
    """
    Performs simulation over multiple timesteps
    """
    f_data = np.zeros(t_steps)
    for i in range(t_steps):
        f_data[i] = func(data)
        data = solver(data)
    return data, f_data


def empty(X_data):
    """
    A dummy function for use with iterator
    """
    return 0.0


@curry
def print_field(X_data, i, steps=100):
    """
    outputs microstructure after particular steps
    """

    if (i+1) % steps == 0:
        np.savetxt(fname=("phi_%s.dat" % (i)), X=X_data)


def error(X, Y):
    n = X.shape[0]*X.shape[1]
    return (np.sum((X-Y)**2))


def FieldConcentration(X,sample_id=0):
    if X.ndim == 3:
        try:
            return np.sum(X[sample_id])
        except IndexError:
            print("Requested Sample # %d exceeds data size" % (sample_id+1))
    else:
        return np.sum(X)
