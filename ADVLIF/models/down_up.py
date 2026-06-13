import torch
import torch.nn as nn
import torch.nn.functional as F


# ADV 1 keeps only the operators used by UNetSegHead_Adv:
# base blocks, reaction-diffusion downsampling, Poisson-guided upsampling,
# and the factory helpers consumed by segheads.py.
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VANDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class VANUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


def grad_n(x):
    gx = torch.roll(x, shifts=-1, dims=3) - x
    gy = torch.roll(x, shifts=-1, dims=2) - x
    gx[..., -1] = 0.0
    gy[:, :, -1, :] = 0.0
    return gx, gy


def div_n(px, py):
    dx = px - torch.roll(px, shifts=1, dims=3)
    dy = py - torch.roll(py, shifts=1, dims=2)
    dx[..., 0] = px[..., 0]
    dy[:, :, 0, :] = py[:, :, 0, :]
    return dx + dy


def lap_n(x):
    x_r = torch.roll(x, 1, dims=3)
    x_l = torch.roll(x, -1, dims=3)
    x_d = torch.roll(x, 1, dims=2)
    x_u = torch.roll(x, -1, dims=2)
    x_r[..., 0] = x[..., 0]
    x_l[..., -1] = x[..., -1]
    x_d[:, :, 0, :] = x[:, :, 0, :]
    x_u[:, :, -1, :] = x[:, :, -1, :]
    return (x_r + x_l + x_d + x_u) - 4.0 * x


class ReactionDiffusionDown(nn.Module):
    def __init__(self, in_ch, out_ch, steps=2, dt=0.2):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1), nn.Softplus())
        self.a = nn.Parameter(torch.zeros(1, in_ch, 1, 1))
        self.b = nn.Parameter(torch.ones(1, in_ch, 1, 1) * 0.1)
        self.post = nn.Conv2d(in_ch, out_ch, 1)
        self.steps, self.dt = steps, dt

    def forward(self, x):
        u = x
        d = self.D(x)
        for _ in range(self.steps):
            u = u + self.dt * (d * lap_n(u) + self.a * u - self.b * (u * u * u))
        return self.post(F.avg_pool2d(u, 2))


class PoissonGuidedUp(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, iters=8, tau=0.25):
        super().__init__()
        self.proj_u = nn.Conv2d(in_ch, out_ch, 1)
        self.proj_s = nn.Conv2d(skip_ch, out_ch, 1)
        self.iters, self.tau = iters, tau

    def forward(self, low, skip):
        size = skip.shape[-2:]
        u = F.interpolate(self.proj_u(low), size=size, mode='bilinear', align_corners=False)
        s = self.proj_s(skip).detach()
        b = div_n(*grad_n(s))
        for _ in range(self.iters):
            r = lap_n(u) - b
            u = u - self.tau * lap_n(r)
        return u


DOWN_REGISTRY = {
    'base': VANDown,
    'rd': ReactionDiffusionDown,
}

UP_REGISTRY = {
    'base': VANUp,
    'poisson': PoissonGuidedUp,
}

BLOCK_REGISTRY = {
    'base': ConvBlock,
}


def make_down(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in DOWN_REGISTRY:
        raise ValueError(f"Unknown down type '{name}'. Available: {list(DOWN_REGISTRY.keys())}")
    return DOWN_REGISTRY[name](in_ch, out_ch, **kwargs)


def make_up(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in UP_REGISTRY:
        raise ValueError(f"Unknown up type '{name}'. Available: {list(UP_REGISTRY.keys())}")
    if name == 'poisson':
        skip_ch = kwargs.pop('skip_ch')
        return UP_REGISTRY[name](in_ch, skip_ch, out_ch, **kwargs)
    return UP_REGISTRY[name](in_ch, out_ch, **kwargs)


def make_block(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type '{name}'. Available: {list(BLOCK_REGISTRY.keys())}")
    return BLOCK_REGISTRY[name](in_ch, out_ch, **kwargs)
