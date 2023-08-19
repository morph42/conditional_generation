import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

from tqdm import tqdm
import numpy as np

from utils.diffusion import GaussianDiffusion 
from utils.unet import UNetModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
num_workers = 8
epochs = 100
dataroot = './data'
modelroot = './model/mnist_unet.pth'
update_every = 100

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(2),
    transforms.Normalize(0.5, 0.5)
])

mnist_train = datasets.MNIST(root='data/', train=True, transform=transforms, download=True)
data_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

net = UNetModel(image_size=32, in_channels=1, out_channels=1, 
                model_channels=64, num_res_blocks=2, channel_mult=(1,2,3,4),
                attention_resolutions=[8,4], num_heads=4).to(device)
net.train()
opt = optim.Adam(net.parameters(), lr=1e-4)
diffusion = GaussianDiffusion(T=1000, schedule='linear')


def train():
    for e in range(epochs):
        print(f'Epoch [{e+1}/{epochs}]')
        losses = []
        batch_bar = tqdm(data_loader)
        for i, batch in enumerate(batch_bar):
            img, labels = batch

            # Sample from the diffusion process
            t = np.random.randint(1, diffusion.T+1, img.shape[0]).astype(int)
            xt, epsilon = diffusion.sample(img, t)
            t = torch.from_numpy(t).float().view(img.shape[0])
            
            # Pass through network
            out = net(xt.float().to(device), t.to(device))

            # Compute loss and backprop
            loss = F.mse_loss(out, epsilon.float().to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            if i % update_every == 0:
                batch_bar.set_postfix({'Loss': np.mean(losses)})
                losses = []
                
        batch_bar.set_postfix({'Loss': np.mean(losses)})
        losses = []

        # Save/Load model
        torch.save(net.state_dict(), modelroot)


if __name__ == '__main__':
    train()
    