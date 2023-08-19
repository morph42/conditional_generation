import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.lenet import LeNet

import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='gray')


batch_size = 128
lr = 1e-3
epochs = 50 
manual_seed = 42
num_workers = 8
modelroot = "models/cnet.pth"
dataroot = "data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(manual_seed)

transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(dataroot, train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = LeNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train():    
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0
        pbar = tqdm.tqdm(dataloader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            pbar.set_description(f"Train Epoch: {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(dataloader.dataset)}] Loss: {loss.item()/len(data):.6f}")
        print(f"====> Epoch: {epoch+1}/{epochs} Average loss: {train_loss/len(dataloader.dataset):.4f}")

        torch.save(model.state_dict(), modelroot)


if __name__ == "__main__":
    train()
    