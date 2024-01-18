import math
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd
import models.conv2d_gradfix as conv2d_gradfix
from torchvision.utils import save_image

def freeze(net: nn.Module):
    for p in net.parameters():
        p.requires_grad_(False)

def unfreeze(net: nn.Module):
    for p in net.parameters():
        p.requires_grad_(True)

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def run_G(z, mixing, g_mapping:nn.Module, g_synthesis:nn.Module):
    ws = g_mapping(z, None)
    if mixing > 0:
        cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        cutoff = torch.where(torch.rand([], device=ws.device) < mixing, cutoff, torch.full_like(cutoff, ws.shape[1]))
        ws[:, cutoff:] = g_mapping(torch.randn_like(z), None, skip_w_avg_update=True)[:, cutoff:]
    img = g_synthesis(ws)
    return img, ws

def save_imgs(z, generator:nn.Module, save_dir_img, epoch=0):
    with torch.no_grad():
        fake = generator(z, None)
    fake = fake.detach()
    fake = (fake * 0.5 + 128 / 255).clamp(0, 1)
    save_image(fake, f"{save_dir_img}/image_epoch_{epoch}.jpg", 
                normalize=False, nrow=4, padding=0)

def save_models(args, generator, discriminator, g_ema, save_name):
    param_dict = {}
    for name, module in [('G', generator), ('D', discriminator), ('G_ema', g_ema)]:
        param_dict[name] = module.state_dict()
    torch.save(param_dict, save_name)

def accumulate(model1:nn.Module, model2:nn.Module, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
