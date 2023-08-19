import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.vae import VAE

from tqdm import tqdm
import matplotlib.pyplot as plt


batch_size = 128
lr = 1e-3
epochs = 50 
manual_seed = 42
num_workers = 8
modelroot = "models/vae.pth"
dataroot = "data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(manual_seed)

transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(dataroot, train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
bce_loss = nn.BCELoss(reduction="sum")


def kld_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train():
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(dataloader)
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            data = data.view(-1, 784)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = bce_loss(recon_batch, data) + kld_loss(mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            pbar.set_description(f"Train Epoch: {epoch+1}/{epochs} [{batch_idx*len(data)}/{len(dataloader.dataset)}] Loss: {loss.item()/len(data):.6f}")
        print(f"====> Epoch: {epoch+1}/{epochs} Average loss: {train_loss/len(dataloader.dataset):.4f}")

        torch.save(model.state_dict(), modelroot)


def plot_img(n_sample):
    model.load_state_dict(torch.load(modelroot))
    model.eval()
    with torch.no_grad():
        num = dataset[0][0]
        num = num.view(-1, 784) 
        num = num.to(device)
        mu, logvar = model.encode(num)
        mu = mu.to(device)
        logvar = logvar.to(device)
        sample = model.sample(mu, logvar, n_sample, device, noise=1e-1)
        sample = sample.view(n_sample, 1, 28, 28)
        save_image(sample, "sample.png")


# if __name__ == "__main__":
#     train()
#     plot_img(16)
