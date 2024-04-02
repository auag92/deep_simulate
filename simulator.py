import torch
import numpy as np
import torch.nn as nn

class CahnHilliard2D:
    """
    A PyTorch module for simulating the reaction-diffusion process using the Cahn-Hilliard equation.

    Attributes:
        gamma (float): A scaling factor in the reaction-diffusion equation.
        dx (float): The spatial discretization step.
        dt (float): The time discretization step.
        device (torch.device): The device (CPU or GPU) on which computations will be performed.
    """

    def __init__(self, gamma=1., dx=0.25, dt=0.001, device=None):
        """
        Initializes the ReactionDiffusion2D module with given simulation parameters.

        Args:
            gamma (float, optional): A scaling factor in the reaction-diffusion equation. Defaults to 1.
            dx (float, optional): The spatial discretization step. Defaults to 0.25.
            dt (float, optional): The time discretization step. Defaults to 0.001.
            device (torch.device, optional): The device (CPU or GPU) for computations. 
                                             Defaults to GPU if available, else CPU.
        """
        self.gamma = gamma
        self.dx = dx
        self.dt = dt
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') if device is None else device

    def forward(self, init_data, n_steps=1):
        """
        Performs the forward pass of the simulation, iterating the reaction-diffusion process.

        Args:
            init_data (numpy.ndarray or torch.Tensor): The initial state of the system.
            n_steps (int, optional): The number of time steps to simulate. Defaults to 1.

        Returns:
            torch.Tensor: The state of the system after n_steps of simulation.
        """
        X = torch.tensor(init_data, dtype=torch.float32).to(self.device)

        for _ in range(n_steps):
            X = self.step(X)

        return X.cpu().numpy()

    def step(self, X):
        """
        Performs a single time step of the reaction-diffusion process.

        Args:
            X (torch.Tensor): The current state of the system.

        Returns:
            torch.Tensor: The updated state of the system after one time step.
        """
        N = X.shape[1]
        L = self.dx * N
        k = np.arange(N)

        # Adjusting for Nyquist frequency
        if N % 2 == 0:
            N1 = N // 2
            N2 = N1
        else:
            N1 = (N - 1) // 2
            N2 = N1 + 1
        k[N2:] = (k - N1)[:N1]
        k = k * 2 * np.pi / L

        i_ = np.indices(X.shape[1:])
        ksq = torch.tensor(np.sum(k[i_] ** 2, axis=0)[None], dtype=torch.float32).to(self.device)

        FX = torch.fft.fftn(X, dim=(1, 2))
        FX3 = torch.fft.fftn(X ** 3, dim=(1, 2))

        A1, A2 = 3., 0.
        explicit = ksq * (A1 - self.gamma * A2 * ksq)
        implicit = ksq * ((1 - A1) - self.gamma * (1 - A2) * ksq)

        Fy = (FX * (1 + self.dt * explicit) - ksq * self.dt * FX3) / (1 - self.dt * implicit)

        X = torch.fft.ifftn(Fy, dim=(1, 2)).real

        return X
