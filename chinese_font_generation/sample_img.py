import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

from torchvision.utils import save_image

from tqdm import tqdm

from utils.diffusion import GaussianDiffusion
from utils.unet import UNetModel
# from utils.custom_dataset import CustomDataset
from utils.style_encode import StyleEncoder


def sample_dm(n_sample, target=[0 for _ in range(32)], method='ddpm'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen_net = UNetModel(image_size=64, in_channels=1, out_channels=1,
                        model_channels=64, num_res_blocks=2, channel_mult=(1, 2, 3, 4),
                        attention_resolutions=[8, 4], num_heads=4).to(device)
    gen_net.load_state_dict(torch.load('models/unet_YingZhangXingShu.pth'))
    gen_net.to(device)
    gen_net.eval()

    class_net = StyleEncoder().to(device)
    class_net.load_state_dict(torch.load('models/FZSuSXSJF_classifier.pth'))
    class_net.to(device)
    class_net.eval()

    epochs = 1000
    target_tensor = torch.tensor(target).to(device)

    if method == 'ddpm':
        diffusion = GaussianDiffusion(T=1000, schedule='linear', gamma=0.5)
    elif method == 'ddim':
        diffusion = GaussianDiffusion(T=1000, schedule='linear', total_steps=1000, implicit=True, gamma=0.5)
    
    x = torch.randn(n_sample, 1, 64, 64).to(device)
    x = torch.nn.Parameter(x.to(device))
    z = x.clone()
    opt = optim.Adam([x], lr=0.1)
    mu_ = torch.zeros_like(x).to(device)
    inner_steps = 10

    for i in tqdm(range(epochs)):
        with torch.no_grad():
            t = epochs - i
            t = np.clip(int(t), 1, diffusion.T)

            alpha_t = diffusion.alpha[t-1]
            beta_t = diffusion.beta[t-1]
            rho = alpha_t / np.sqrt(beta_t)
            if i > 0:
                mu_ = mu_ + rho*(x-z)
            z = x + mu_ / rho
            z = z.clamp(-1, 1)
            z = diffusion.inverse(gen_net, z, t).to(device)

        for _ in range(inner_steps):
            opt.zero_grad()
            fx = class_net(x) - target_tensor
            loss_fn = (mu_*(x-z)).sum() + 0.5*rho * \
                ((x-z)**2).sum() + 1e2 * (fx ** 2).sum()
            loss_fn.backward()
            opt.step()

    return x.detach().cpu().numpy(), z.detach().cpu().numpy()


if __name__ == '__main__':
    x, z = sample_dm(16, 8)
    save_image(torch.from_numpy(x), 'imgs/ddpm_x.png', nrow=4, normalize=True)
    save_image(torch.from_numpy(z), 'imgs/ddpm_z.png', nrow=4, normalize=True)
    