from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from utils.style_encode import StyleEncoder
from utils.custom_dataset import CustomDataset


def train_process(
    image_size,
    dataroot,
    batch_size,
    epochs,
    lr,
    sty_dim,
    modelroot,
    num_workers,
    device,
    method,
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Grayscale(),
        ]
    )
    dataset = CustomDataset("font_files/3500_style.txt", dataroot, transform=transform)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    net = StyleEncoder(sty_dim=sty_dim).to(device)
    criterion = F.mse_loss

    if method == "Adam":
        opt = optim.Adam(net.parameters(), lr=lr)
    elif method == "SGD":
        opt = optim.SGD(net.parameters(), lr=lr)
    else:
        raise ValueError("No such optimizer")

    net.train()
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        pbar = tqdm(dataloader)
        pbar.set_description("Epoch {}".format(epoch))
        losses = []

        for i, (imgs, temp_labels) in enumerate(pbar):
            imgs = imgs.to(device)
            temp = torch.zeros(32, temp_labels[0].shape)
            for j in range(32):
                temp[j] = temp_labels[j]
            labels = temp.to(device)
            labels = torch.transpose(labels, 0, 1)
            # temp = [label.tolist() for label in temp_labels]
            # labels = torch.tensor(temp, dtype=torch.long).to(device)
            # labels = torch.transpose(labels, 0, 1)

            opt.zero_grad()
            out = net(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pbar.set_description("Epoch {} Loss {:.4f}".format(epoch, np.mean(losses)))

        torch.save(net.state_dict(), modelroot)


def main():
    args = ArgumentParser()
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--image_size", type=int, default=64)
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--sty_dim", type=int, default=32)
    args.add_argument("--dataroot", type=str, default="data/")
    args.add_argument(
        "--modelroot", type=str, default="models/YingZhangXingShu_classifier.pth"
    )
    args.add_argument("--num_workers", type=int, default=16)
    args.add_argument("--device", type=str, default="cpu")
    args.add_argument("--method", type=str, default="Adam")
    args = args.parse_args()

    train_process(
        image_size=args.image_size,
        dataroot=args.dataroot,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        sty_dim=args.sty_dim,
        modelroot=args.modelroot,
        num_workers=args.num_workers,
        device=args.device,
        method=args.method,
    )


if __name__ == "__main__":
    main()
