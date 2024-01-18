import argparse
import os
import logging
import pickle
import math
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch.cuda
import torch.backends.cudnn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data import DataLoader
from torch import nn, optim, autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets.load import get_dataset, get_celebahq, get_tsinghuadogs
from models.non_leaking import augment, AdaptiveAugment
from util.fid import load_patched_inception_v3, extract_features, calc_fid
from util.system import init_dist, init_log
from util.ada import *
from util.attack import init_attack, adv_loss

# fixed stylegan params
r1_reg = 10
path_reg = 2
ada_target = 0.6
ada_length = 500 * 1000
g_reg_every = 4
d_reg_every = 16
mixing = 0.9 # probability of latent code mixing
path_batch_shrink = 2 # batch size reducing factor for 
                      # the path length regularization 
                      # (reduce memory consumption)

def get_fid(generator, train_loader, save_dir_model):
    inception = load_patched_inception_v3().cuda()
    inception.eval()
    # get real features
    if os.path.exists(f'{save_dir_model}/inception.npz'):
        if dist.get_rank() == 0:
            loaded_param = np.load(f'{save_dir_model}/inception.npz')
            real_mean = loaded_param['mean']
            real_cov = loaded_param['cov']
    else:
        features_dist = extract_features(train_loader, inception)
        # merge
        features = [torch.zeros_like(features_dist) for _ in range(args.world_size)]
        dist.all_gather(features, features_dist)
        features = torch.cat(features, dim=0).cpu().numpy()
        if dist.get_rank() == 0:
            real_mean = np.mean(features, 0)
            real_cov = np.cov(features, rowvar=False)
            np.savez(f'{save_dir_model}/inception.npz',
                     mean=real_mean,
                     cov=real_cov,
                     features=features)
    # get sample features
    num_samples = 30000
    bs = 100
    with torch.no_grad():
        features_dist = []
        z = torch.rand(num_samples // args.world_size, args.z_dim, device='cuda')
        pbar = range(math.ceil(z.shape[0] / bs))
        if dist.get_rank() == 0:
            pbar = tqdm(pbar)
        for i in pbar:
            z_batch = z[i * bs : (i + 1) * bs]
            imgs = generator(z_batch, c=None)
            feature = inception(imgs)[0].view(imgs.shape[0], -1)
            features_dist.append(feature)
        features_dist = torch.cat(features_dist, dim=0)

    # merge
    features = [torch.zeros_like(features_dist) for _ in range(args.world_size)]
    dist.all_gather(features, features_dist)
    features = torch.cat(features, dim=0).cpu().numpy()
    fid = np.inf
    if dist.get_rank() == 0:
        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)
        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    dist.barrier()
    return fid

def get_loader(args):
    # load aux dataset
    if args.aux == 'celebahq':
        train = get_celebahq(
            resolution=args.resolution,
            normalize=True,
        )
    elif args.aux == 'tsinghuadogs':
        train = get_tsinghuadogs(
            resolution=args.resolution,
            crop=args.crop,
            normalize=True,
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_loader = DataLoader(
        train,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    if dist.get_rank() == 0:
        print(f'finish loading {args.aux} dataset')
    return train_loader, train_sampler

def init_training(args, load_dir_gan):
    # ada_every = 256
    ema_kimg = 10 # Half-life of the exponential moving average (EMA) of generator weights.

    # load stylegan
    with open(load_dir_gan, 'rb') as f:
        loaded_param = pickle.load(f)
        generator: nn.Module = loaded_param['G_ema'].cuda()
        discriminator: nn.Module = loaded_param['D'].cuda()
    unfreeze(generator)
    unfreeze(discriminator)
    g_mapping = DDP(generator.mapping, device_ids=[args.gpu], broadcast_buffers=False)
    g_synthesis = DDP(generator.synthesis, device_ids=[args.gpu], broadcast_buffers=False)
    discriminator = DDP(discriminator, device_ids=[args.gpu], broadcast_buffers=False)

    # ema
    g_ema = deepcopy(generator)
    ema_nimg = ema_kimg * 1000
    ema_beta = 0.5 ** (args.batch_size * args.world_size / max(ema_nimg, 1e-8))

    # optimizer
    g_reg_ratio = g_reg_every / (g_reg_every + 1)
    d_reg_ratio = d_reg_every / (d_reg_every + 1)
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
    )

    ada_augment = AdaptiveAugment(ada_target, ada_length, 8, 'cuda')
    return g_mapping, g_synthesis, generator, discriminator, \
            g_optim, d_optim, g_ema, ema_beta, ada_augment

def main(args):
    args = init_dist(args)
    logger = init_log(args)

    sl_dir_model = f'./.saved_models/seed_{args.seed}/{args.dataset}/{args.target}'
    save_dir_img = f'./.saved_figs/seed_{args.seed}/{args.dataset}/{args.target}/' + \
                   f'{args.attack}_{args.epochs}_transfer'
    if dist.get_rank() == 0:
        # create save dir and check load dir
        for directory in [save_dir_img, sl_dir_model]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    dist.barrier()
    load_dir_gan = f'./pretrained/transfer-learning-source-nets/' + \
        f'celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl'
    if args.dataset == 'stanforddogs':
        load_dir_gan = f'./pretrained/transfer-learning-source-nets/' + \
        f'lsundog-res256-paper256-kimg100000-noaug.pkl'

    # get data loader
    train_loader, train_sampler = get_loader(args)

    # initialize training
    g_mapping, g_synthesis, generator, discriminator, \
    g_optim, d_optim, g_ema, ema_beta, ada_augment = \
                                                        init_training(args, load_dir_gan)

    # init attack
    target, resize, criterion, eigvals, eigvects = \
                                    init_attack(args, sl_dir_model, generator)

    # fid, fixed noise and evaluate at epoch 0
    fid = get_fid(generator, train_loader, sl_dir_model)
    if dist.get_rank() == 0:
        logger.info(f'---------------------' + \
                    f'transfer stylegan with {args.aux} and {args.target} {args.dataset}' + \
                    f'---------------------')
        logger.info(f'epoch 0; fid: {fid}')
        z_fix = torch.rand(8, args.z_dim, device='cuda')
        save_imgs(z_fix, generator, save_dir_img, epoch=0)
    dist.barrier()

    # start training
    step = 0
    augment_p = 0.
    mean_path_length = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        for real, _ in tqdm(train_loader):
            real = real.cuda()
            bs = real.size(0)
            step += 1

            # train discriminator
            freeze(generator)
            unfreeze(discriminator)
            z = torch.rand(bs, args.z_dim, device='cuda')
            fake, _ = run_G(z, mixing, g_mapping, g_synthesis)
            fake_aug, _ = augment(fake, augment_p)
            real_aug, _ = augment(real, augment_p)
            logits_real = discriminator(real_aug, None)
            logits_fake = discriminator(fake_aug, None)
            loss_d = d_logistic_loss(logits_real, logits_fake)
            d_optim.zero_grad()
            loss_d.backward()
            d_optim.step()

            # tune aug_p
            augment_p = ada_augment.tune(logits_real)

            # regularize d
            if step % d_reg_every == 0:
                real.requires_grad = True
                real_aug, _ = augment(real, augment_p)
                logits_real = discriminator(real_aug, None)
                loss_r1 = d_r1_loss(logits_real, real)
                d_optim.zero_grad()
                (r1_reg / 2 * loss_r1 * d_reg_every + 0 * logits_real[0]).backward()
                d_optim.step()

            # train g
            freeze(discriminator)
            unfreeze(generator)

            z = torch.rand(bs, args.z_dim, device='cuda')
            fake, _ = run_G(z, mixing, g_mapping, g_synthesis)
            fake_aug, _ = augment(fake, augment_p)
            logits_fake = discriminator(fake_aug, None)

            loss_g = g_nonsaturating_loss(logits_fake)
            loss_g = adv_loss(args, fake, fake_aug, resize, target, criterion,
                              eigvals, eigvects, loss_g)

            g_optim.zero_grad()
            loss_g.backward()
            g_optim.step()

            # regularize g
            if step % g_reg_every == 0:
                path_batch_size = max(1, args.batch_size // path_batch_shrink)
                z = torch.rand(path_batch_size, args.z_dim, device='cuda')
                fake, latent = run_G(z, mixing, g_mapping, g_synthesis)
                loss_path, mean_path_length, path_lengths= g_path_regularize(
                    fake, latent, mean_path_length
                )
                g_optim.zero_grad()
                loss_weighted_path = path_reg * g_reg_every * loss_path
                if path_batch_shrink:
                    loss_weighted_path += 0 * fake[0, 0, 0, 0]
                loss_weighted_path.backward()
                g_optim.step()

            # update ema
            accumulate(g_ema, generator, ema_beta)

        # calculate fid, save images and models
        fid = get_fid(generator, train_loader, sl_dir_model)
        if dist.get_rank() == 0:
            logger.info(f'epoch {epoch}; fid: {fid}')
            save_name = f'{sl_dir_model}/{args.attack}_{args.epochs}_gan.pth'
            save_models(args, generator, discriminator, g_ema, save_name)
            save_imgs(z_fix, g_ema, save_dir_img, epoch)
        dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp', type=int, default=1)

    # model and dataset setting
    parser.add_argument('--target', type=str, default='vgg16')
    parser.add_argument('--aux', type=str, default='celebahq')
    parser.add_argument('--dataset', type=str, default='umdfaces')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--attack', type=str, default='vanilla')

    # optimization setting
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10) # 200)
    parser.add_argument('--early_start', type=int, default=5, help='# of early start epochs for T')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers for data loaders')
    parser.add_argument('--pin_memory', type=int, default=1, help='pin the memory for train loader')

    args = parser.parse_args()
    main(args)