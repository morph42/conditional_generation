import numpy as np
import torch
import math

class GaussianDiffusion():
    '''Gaussian Diffusion process with linear beta scheduling'''
    def __init__(self, T, schedule, total_steps=1000, implicit=False, gamma=0.0):
        # Diffusion steps
        self.T = T
    
        # Noise schedule
        if schedule == 'linear':
            b0=1e-4
            bT=2e-2
            self.beta = np.linspace(b0, bT, total_steps)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T+1, 1)) / self.__cos_noise(0) # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

        self.betabar = np.cumprod(self.beta)
        self.alpha = 1 - self.beta
        self.alphabar = np.cumprod(self.alpha)       
        if implicit == False:
            self.step = np.array(range(0, total_steps+1))
            assert T == total_steps, 'Steps mismatch'
        else:
            self.interval = total_steps // T
            self.step = np.array(range(0, total_steps+1, self.interval))

        self.implicit = implicit
        self.gamma = gamma

    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t/self.T + offset) / (1+offset)) ** 2
   
    def sample(self, x0, t):        
        # Select noise scales
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))       
        atbar = torch.from_numpy(self.alphabar[t-1]).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'
        
        # Sample noise and add to x0
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1-atbar) * epsilon 
        return xt, epsilon

    def inverse(self, net, x, t0, device='cuda:0'):
        t = self.step[t0-1]
        at = self.alpha[t]
        atbar = self.alphabar[t]

        with torch.no_grad():
            t = torch.tensor([t]).view(1)
            pred = net(x, t.float().to(device))

        if t0 > 1:
            t_prev = self.step[t0-2]
            z = torch.randn_like(x)
            atbar_prev = self.alphabar[t_prev]
            beta_tilde = self.beta[t] * (1 - atbar_prev) / (1 - atbar) 
        else:
            z = torch.zeros_like(x)
            atbar_prev = 1
            beta_tilde = 0

        if self.implicit == False: # DDPM sampling
            x = (1 / np.sqrt(at)) * (x - ((1-at) / np.sqrt(1-atbar)) * pred).clamp(-1,1) + self.gamma*np.sqrt(beta_tilde) * z
        elif self.implicit == True: # DDIM sampling
            f_theta = (1 / np.sqrt(atbar)) * (x - np.sqrt(1-atbar)*pred).clamp(-1,1)
            x = np.sqrt(atbar_prev)*f_theta + np.sqrt(1-atbar_prev-self.gamma**2*beta_tilde)*pred + self.gamma*np.sqrt(beta_tilde)*z
        return x    
      
    def total_inverse(self, net, shape=(1,64,64), steps=None, x=None, start_t=None, device='cpu'):
        # Specify starting conditions and number of steps to run for 
        if x is None:
            x = torch.randn((1,) + shape).to(device)
        if start_t is None:
            start_t = self.T
        if steps is None:
            steps = self.T

        for t in range(start_t, start_t-steps, -1):
            at = self.alpha[t-1]
            atbar = self.alphabar[t-1]
            
            if t > 1:
                z = torch.randn_like(x)
                atbar_prev = self.alphabar[t-2]
                beta_tilde = self.beta[t-1] * (1 - atbar_prev) / (1 - atbar) 
            else:
                z = torch.zeros_like(x)
                beta_tilde = 0

            with torch.no_grad():
                t = torch.tensor([t]).view(1)
                pred = net(x, t.float().to(device))

            x = (1 / np.sqrt(at)) * (x - ((1-at) / np.sqrt(1-atbar)) * pred) + np.sqrt(beta_tilde) * z

        return x    
    
    def inverse_DDIM(self, net, x, t, device='cuda:0'):
        """
        This applies the unconditional sampling of the diffusion step using the
        DDIM method: https://arxiv.org/abs/2010.02502
        f_theta acts as an approximation for x_0, and the rest follows equation
        (7) in the paper. For DDIM, we have that std_dev = 0
        This solves the problem of stochasticity, and it is supposed to be 10x
        to 100x quicker than the DDPM method
        """

        with torch.no_grad():
            t = torch.tensor([t]).view(1)
            pred = net(x, t.float().to(device))

        den = 1 / torch.sqrt(self.alphabar[t-1])
        f_theta = (x - torch.sqrt(1 - self.alphabar[t-1]) * pred) * den

        if t > 1:
            part1 = torch.sqrt(self.alphabar[t-2]) * f_theta
            part2 = torch.sqrt(1-self.alphabar[t-2])
            den = 1 / torch.sqrt(1-self.alphabar[t-1])
            scale = (x - torch.sqrt(self.alphabar[t-1]) * f_theta) * den
            x = part1 + part2 * scale
        else:
            x = f_theta

        return x
    
