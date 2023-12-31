import torch
import torch.nn as nn 
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def sample(self, mu, logvar, n_sample, device, noise = 1e-2):
        with torch.no_grad():
            mu_ = mu.repeat(n_sample, 1)
            logvar_ = logvar.repeat(n_sample, 1)
            mu_ = mu_.to(device)
            logvar_ = logvar_.to(device)
            mu_ += torch.randn_like(mu_).to(device) * noise
            logvar_ += torch.randn_like(logvar_).to(device) * noise
            z = self.reparameterize(mu_, logvar_)

            return self.decode(z)
        