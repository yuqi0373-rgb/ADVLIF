import torch
import torch.nn as nn
import torch.nn.functional as F

from .down_up import make_down, make_up, make_block


MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------
# ADV 1 only
# ---------------------------
def dice_loss(logits, target, eps=1e-6):
    # logits: (N,1,H,W), target: (N,1,H,W) in {0,1}
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3))
    den = (prob + target).sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def kl_div_with_logits(p_logits, q_logits):
    # KL(p || q) on pixel-wise Bernoulli via logits.
    p = torch.sigmoid(p_logits).clamp(1e-6, 1 - 1e-6)
    q = torch.sigmoid(q_logits).clamp(1e-6, 1 - 1e-6)
    return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean()


def tv_loss(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


class ALGate(nn.Module):
    """
    Clean gate g = sigma(phi([f_m; f_a])). During supervised training, an
    adversarial copy of g is built with FGSM steps and used for KL consistency.
    """
    def __init__(self, ch, reduction=4, eps=0.1, steps=1):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.eps, self.steps = eps, steps
        self.phi = nn.Sequential(
            nn.Conv2d(2 * ch, mid, 1), nn.GELU(),
            nn.Conv2d(mid, ch, 1),
        )
        self.surr = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, f_m, f_a, y=None):
        z = torch.cat([f_m, f_a], 1)
        g = torch.sigmoid(self.phi(z))
        fused = g * f_m + (1 - g) * f_a

        kl_robust = fused.new_zeros(())
        smooth = tv_loss(g)
        if (y is not None) and self.steps > 0:
            g_adv = g.clone().detach().requires_grad_(True)
            for _ in range(self.steps):
                f_adv = g_adv * f_m + (1 - g_adv) * f_a
                logits_surr = self.surr(f_adv)
                loss_sur = dice_loss(logits_surr, y) + bce_loss(logits_surr, y)
                grad = torch.autograd.grad(loss_sur, g_adv, retain_graph=False, create_graph=False)[0]
                g_adv = (g_adv + self.eps * grad.sign()).clamp(0.0, 1.0).detach().requires_grad_(True)

            with torch.no_grad():
                f_clean = fused
                f_worst = g_adv * f_m + (1 - g_adv) * f_a
            kl_robust = kl_div_with_logits(self.surr(f_clean), self.surr(f_worst))

        return fused, g, {"gate_tv": smooth, "gate_robust": kl_robust}


class ARRSkip(nn.Module):
    """
    Aux-conditioned routing mask r blends identity and residual skip features.
    During training, an adversarial router copy provides a consistency loss.
    """
    def __init__(self, ch, reduction=4, eps=0.1):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.eps = eps
        self.router = nn.Sequential(nn.Conv2d(ch, mid, 1), nn.GELU(), nn.Conv2d(mid, ch, 1))
        self.residual = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 1))

    def forward(self, s_main, a_aux, head_for_grad=None):
        if a_aux.shape[-2:] != s_main.shape[-2:]:
            a_aux = F.interpolate(a_aux, size=s_main.shape[-2:], mode='bilinear', align_corners=False)

        r = torch.sigmoid(self.router(a_aux))
        s_clean = (1 - r) * s_main + r * self.residual(s_main)

        cons_loss = s_clean.new_zeros(())
        if head_for_grad is not None and self.training:
            with torch.enable_grad():
                r_adv = r.clone().detach().requires_grad_(True)
                s_adv = (1 - r_adv) * s_main + r_adv * self.residual(s_main)
                logits_clean = head_for_grad(s_clean).detach()
                logits_adv = head_for_grad(s_adv)
                loss = kl_div_with_logits(logits_clean, logits_adv)
                grad = torch.autograd.grad(loss, r_adv, retain_graph=False, create_graph=False)[0]

                r_adv = (r_adv + self.eps * grad.sign()).clamp(0.0, 1.0).detach()
                s_adv = (1 - r_adv) * s_main + r_adv * self.residual(s_main)
                cons_loss = kl_div_with_logits(logits_clean, head_for_grad(s_adv))

        return s_clean, {"arr_cons": cons_loss, "arr_tv": tv_loss(r)}


@register_model('unet_adv')
class UNetSegHead_Adv(nn.Module):
    """
    ADV 1 segmentation head.

    Inputs:
      x: (N,15,H,W), channels 0:12 are aux and 12:15 are main.
      y: (N,1,H,W), optional binary mask for adversarial regularization.
    """
    def __init__(self, base=32, depth=(2, 2, 2, 1), k=3, gate_eps=0.1, gate_steps=1, arr_eps=0.1,
                 w_dice=1.0, w_bce=1.0, w_gate_tv=1e-3, w_gate_rob=1e-3, w_arr_cons=1e-3, w_arr_tv=5e-4):
        super().__init__()

        self.inc_aux = make_block('base', 12, base)
        self.inc_main = make_block('base', 3, base)

        self.gate = ALGate(base, eps=gate_eps, steps=gate_steps)

        self.down1 = make_down('base', base, base * 2)
        self.down2 = make_down('base', base * 2, base * 4)
        self.down3 = make_down('base', base * 4, base * 8)
        self.down4 = make_down('rd', base * 8, base * 16, steps=1, dt=0.2)

        self.aux1 = make_down('base', base, base * 2)
        self.aux2 = make_down('base', base * 2, base * 4)
        self.aux3 = make_down('base', base * 4, base * 8)
        self.aux4 = make_down('rd', base * 8, base * 16, steps=1, dt=0.2)

        self.up1 = make_up('poisson', base * 16, base * 8, skip_ch=base * 8, iters=1, tau=0.25)
        self.up2 = make_up('base', base * 8, base * 4)
        self.up3 = make_up('base', base * 4, base * 2)
        self.up4 = make_up('base', base * 2, base)

        self.arr4 = ARRSkip(base * 8, eps=arr_eps)
        self.arr3 = ARRSkip(base * 4, eps=arr_eps)
        self.arr2 = ARRSkip(base * 2, eps=arr_eps)
        self.arr1 = ARRSkip(base, eps=arr_eps)

        self.skip_head4 = nn.Conv2d(base * 8, 1, 1)
        self.skip_head3 = nn.Conv2d(base * 4, 1, 1)
        self.skip_head2 = nn.Conv2d(base * 2, 1, 1)
        self.skip_head1 = nn.Conv2d(base, 1, 1)

        self.outc = nn.Conv2d(base, 1, 1)

        self.w_dice, self.w_bce = w_dice, w_bce
        self.w_gate_tv, self.w_gate_rob = w_gate_tv, w_gate_rob
        self.w_arr_cons, self.w_arr_tv = w_arr_cons, w_arr_tv

    def forward(self, x, y=None, compute_loss=True):
        assert x.ndim == 4 and x.size(1) >= 15
        xa, xm = x[:, :12], x[:, 12:15]

        f_aux0 = self.inc_aux(xa)
        f_main0 = self.inc_main(xm)
        fused, g, gate_loss = self.gate(f_main0, f_aux0, y=y if compute_loss else None)

        e2 = self.down1(fused)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        a2 = self.aux1(f_aux0)
        a3 = self.aux2(a2)
        a4 = self.aux3(a3)
        _ = self.aux4(a4)

        s4, arr4_loss = self.arr4(e4, a4, head_for_grad=self.skip_head4 if compute_loss else None)
        y_ = self.up1(e5, s4)

        s3, arr3_loss = self.arr3(e3, a3, head_for_grad=self.skip_head3 if compute_loss else None)
        y_ = self.up2(y_, s3)

        s2, arr2_loss = self.arr2(e2, a2, head_for_grad=self.skip_head2 if compute_loss else None)
        y_ = self.up3(y_, s2)

        s1, arr1_loss = self.arr1(fused, f_aux0, head_for_grad=self.skip_head1 if compute_loss else None)
        y_ = self.up4(y_, s1)

        logits = self.outc(y_)

        if not compute_loss or (y is None):
            return logits, g, {"gate_name": 'adv'}

        l_gate = self.w_gate_tv * gate_loss["gate_tv"] + self.w_gate_rob * gate_loss["gate_robust"]
        l_arr = self.w_arr_cons * (
            arr1_loss["arr_cons"] + arr2_loss["arr_cons"] + arr3_loss["arr_cons"] + arr4_loss["arr_cons"]
        ) + self.w_arr_tv * (
            arr1_loss["arr_tv"] + arr2_loss["arr_tv"] + arr3_loss["arr_tv"] + arr4_loss["arr_tv"]
        )
        l_total = l_gate + l_arr
        return logits, g, {"total": l_total, "gate_name": 'adv'}


MODEL_Registry = {
    'unet_adv': UNetSegHead_Adv,
}
