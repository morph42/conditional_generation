from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# from torchvision.datasets import ImageFolder
from torchvision import transforms
# from torchvision.utils import save_image
from tqdm import tqdm

from utils.diffusion import GaussianDiffusion
from utils.unet import UNetModel
from utils.custom_dataset import CustomDataset


# device = 'cuda' if torch.cuda.is_available() else 'cpu'


# transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
# dataset = ImageFolder(root='data', transform=transform)
# data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16)


class DiffusionTrain:
    def __init__(
        self,
        device="cuda",
        transform=None,
        dataroot="data/",
        modelroot="models/unet_FZSuSXSJF.pth",
        opt_method="Adam",
        lr=1e-4,
        batch_size=64,
        num_workers=16,
        image_size=64,
        manual_seed=42,
        T=1000,  # number of steps
        schedule="linear",
        epochs=50,
    ):
        self.device = device
        self.image_size = image_size

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    # transforms.Grayscale(),
                ]
            )
        self.transform = transform

        self.dataset = CustomDataset(
            "font_files/3500_style.txt", dataroot, transform=transform
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.opt_method = opt_method
        self.lr = lr
        self.T = T
        self.schedule = schedule
        self.epochs = epochs
        self.modelroot = modelroot

    def train_process(self):
        net = UNetModel(
            image_size=self.image_size,
            in_channels=1,
            out_channels=1,
            model_channels=64,
            num_res_blocks=2,
            channel_mult=(1, 2, 3, 4),
            attention_resolutions=[8, 4],
            num_heads=4,
        ).to(self.device)
        net.train()
        print("Network parameters:", sum([p.numel() for p in net.parameters()]))

        if self.opt_method == "Adam":
            opt = optim.Adam(net.parameters(), lr=self.lr)
        elif self.opt_method == "SGD":
            opt = optim.SGD(net.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError
        diffusion = GaussianDiffusion(T=self.T, schedule=self.schedule)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            losses = []
            pbar = tqdm(self.data_loader)
            for i, batch in enumerate(pbar):
                img, _ = batch
                img = img.to(self.device)

                # sample from the diffusion process
                t = np.random.randint(1, diffusion.T + 1, img.shape[0]).astype(int)
                xt, epsilon = diffusion.sample(img, t)
                t = torch.from_numpy(t).float().view(img.shape[0])

                # Pass through network
                out = net(xt.float().to(self.device), t.to(self.device))

                # Compute loss and backprop
                loss = F.mse_loss(out, epsilon.float().to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())
                pbar.set_description(f"Loss: {np.mean(losses):.4f}")

            # Save model
            torch.save(net.state_dict(), self.modelroot)

    # def sample_imgs(
    #     self, img_path="imgs/", n_samples=16, input_imgs=None, input_t=None
    # ):
    #     net = UNetModel(
    #         image_size=self.image_size,
    #         in_channels=1,
    #         out_channels=1,
    #         model_channels=64,
    #         num_res_blocks=2,
    #         channel_mult=(1, 2, 3, 4),
    #         attention_resolutions=[8, 4],
    #         num_heads=4,
    #     ).to(self.device)
    #     net.load_state_dict(torch.load(self.modelroot))
    #     net.eval()

    #     if input_t is None:
    #         diffusion = GaussianDiffusion(T=self.T, schedule=self.schedule)
    #     else:
    #         diffusion = GaussianDiffusion(T=input_t, schedule=self.schedule)

    #     if input_imgs is None:
    #         with torch.no_grad():
    #             x = diffusion.inverse(
    #                 net=net, shape=(1, 64, 64), device=self.device, n_samples=n_samples
    #             )
    #             save_image(x, img_path + "sample.png")
    #     else:
    #         with torch.no_grad():
    #             x = diffusion.inverse(
    #                 net=net,
    #                 shape=(1, 64, 64),
    #                 device=self.device,
    #                 x=input_imgs,
    #                 start_t=input_t,
    #                 steps=input_t,
    #                 n_samples=input_imgs.shape[0],
    #             )
    #             save_image(x, img_path + "sample.png")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument("--dataroot", type=str, default="data/")
    args.add_argument("--modelroot", type=str, default="models/unet_FZSuSXSJF.pth")
    args.add_argument("--opt_method", type=str, default="Adam")
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--num_workers", type=int, default=16)
    args.add_argument("--image_size", type=int, default=64)
    args.add_argument("--manual_seed", type=int, default=42)
    args.add_argument("--T", type=int, default=1000)
    args.add_argument("--schedule", type=str, default="linear")
    args.add_argument("--epochs", type=int, default=100)
    args = args.parse_args()

    diffusion_train = DiffusionTrain(
        device=args.device,
        dataroot=args.dataroot,
        modelroot=args.modelroot,
        opt_method=args.opt_method,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        manual_seed=args.manual_seed,
        T=args.T,
        schedule=args.schedule,
        epochs=args.epochs,
    )
    # diffusion_train.train_process()
    # diffusion_train.sample_imgs()
