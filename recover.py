import argparse
import os
import math
from tqdm import tqdm
import torch
import torch.cuda
import torch.distributed as dist
from torch import nn, optim
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.utils import save_image
import dnnlib
from torchvision.models import wide_resnet50_2, inception_v3
from models.target import vgg16_bn, resnet50, resnet101, resnet152, alexnet
from models.evolve import IR_50, IR_101, IR_152, IR_SE_101, IR_SE_152, IR_SE_50
from models.stylegan import make_stylegan_kwargs
from datasets.load import get_dataset
from util.ppa import create_target_vector, find_initial_w, poincare_loss, perform_final_selection
from util.system import init_dist, init_log
from util.ada import freeze

def init_attack(args, load_dir_model):
    # get dataset index information
    train, _ = get_dataset(subset='target', name=args.dataset)
    idx_to_class = {v: k for k, v in train.class_to_idx.items()}

    nc = 1000
    if args.dataset == 'stanforddogs':
        nc = 120
    # load target model
    if args.target == 'vgg16':
        target = vgg16_bn(num_classes=nc).cuda()
    elif args.target == 'resnet50':
        target = resnet50(num_classes=nc).cuda()
    elif args.target == 'irse50':
        target = IR_SE_50(num_classes=nc).cuda()
    elif args.target == 'alexnet':
        target = alexnet(num_classes=nc).cuda()
    loaded_param = torch.load(f'{load_dir_model}/target.pth')
    target.load_state_dict(loaded_param)
    target.eval()

    # load gan
    load_name = f'{load_dir_model}/{args.attack}_{args.gan_epochs}_gan.pth'
    loaded_param = torch.load(load_name)
    args = make_stylegan_kwargs(args, cfg='paper256')
    common_kwargs = dict(c_dim=0, img_resolution=256, img_channels=3)
    generator:nn.Module = dnnlib.util.construct_class_by_name(**args.G_kwargs, **common_kwargs).cuda()
    generator.load_state_dict(loaded_param['G_ema'])
    for p in generator.parameters():
        p.requires_grad_(False)

    # create transformation
    if args.dataset == 'umdfaces':
        crop = 200
    elif args.dataset == 'stanforddogs':
        crop = 256
    transform = transforms.Compose([
        transforms.CenterCrop(crop), # 256 -> 200
        transforms.Resize((args.resolution)), # 200 -> 112
    ])

    return generator, target, transform, idx_to_class

def optimize(args, w_dist, labels_dist, synthesis, target, transform):
    # optimize
    optimizer = optim.Adam([w_dist.requires_grad_()], lr=args.lr, betas=args.betas)
    pbar = range(1, args.epochs + 1)
    if dist.get_rank() == 0:
        pbar = tqdm(pbar)
    for epoch in pbar:
        imgs = synthesis(w_dist, noise_mode='const', force_fp32=True)
        logits = target(transform(imgs))
        loss = poincare_loss(logits, labels_dist)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train(args, synthesis, target, transform, w, labels, id):
    freeze(synthesis)
    freeze(target)
    # optimize the samples
    w_optimized = []
    for i in range(math.ceil(w.shape[0] / args.batch_size)):
        w_batch = w[i * args.batch_size : (i + 1) * args.batch_size]
        labels_batch = labels[i * args.batch_size : (i + 1) * args.batch_size]

        # distribute
        dist_len = args.batch_size // args.world_size
        w_dist = w_batch[args.gpu * dist_len : (args.gpu + 1) * dist_len]
        labels_dist = labels_batch[args.gpu * dist_len : (args.gpu + 1) * dist_len]

        # train
        optimize(args, w_dist, labels_dist, synthesis, target, transform)

        # merge
        merge = [torch.zeros_like(w_dist) for _ in range(args.world_size)]
        dist.all_gather(merge, w_dist)
        merge = torch.cat(merge, dim=0)

        w_optimized.append(merge.detach())
        if dist.get_rank() == 0:
            print(f'finish optimizing label {id} batch {i}')
    w_optimized = torch.cat(w_optimized, dim=0)
    return w_optimized.detach()

def save_results(synthesis, transform, w_final, labels, save_dir_img, idx_to_class):
    bs = 50
    for i in range(math.ceil(w_final.shape[0] / bs)):
        w_batch = w_final[i * bs : (i + 1) * bs].cuda()
        labels_batch = labels[i * bs : (i + 1) * bs].cuda()
        with torch.no_grad():
            imgs = synthesis(w_batch, noise_mode='const', force_fp32=True)
            imgs = transform(imgs)
            imgs = (imgs * 0.5 + 128 / 255).clamp(0, 1)
        for j in range(imgs.size(0)):
            key = int(labels_batch[j])
            img = imgs[j]
            folder_name = f'{save_dir_img}/{idx_to_class[key]}'
            save_name = f'{folder_name}/{i * bs + j}.jpg'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            save_image(img.detach(), save_name)

def run(args, i, generator, target, transform, save_dir_img, idx_to_class):
    labels = create_target_vector(
        target_classes=i,
        num_candidates=args.num_candidates,
    )

    # init w
    w = find_initial_w(
        args=args,
        generator=generator,
        target_model=target,
        targets=labels,
        transform=transform,
        truncation_psi=args.trunc_p,
        truncation_cutoff=args.trunc_c,
    )
    synthesis = generator.synthesis

    # train w
    w_optimized = train(args, synthesis, target, transform, w, labels, id=i)
    del w

    # final selection
    w_final, labels = perform_final_selection(
        args=args,
        batch_size=args.batch_size,
        w=w_optimized,
        synthesis=synthesis,
        targets=labels,
        target_model=target,
        samples_per_target=args.num_final
    )
    del w_optimized

    # save
    if dist.get_rank() == 0:
        save_results(synthesis, transform, w_final, labels, save_dir_img, idx_to_class)
    dist.barrier()

def main(args):
    args = init_dist(args)

    load_dir_model = f'./.saved_models/seed_{args.seed}/{args.dataset}/{args.target}'
    save_dir_img = f'./.saved_figs/seed_{args.seed}/{args.dataset}/{args.target}/' + \
                    f'{args.attack}_{args.gan_epochs}_recover'
    if dist.get_rank() == 0:
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)
    # get logger
    logger = init_log(args)
    dist.barrier()

    generator, target, transform, idx_to_class = init_attack(args, load_dir_model)

    for i in range(args.num_classes):
        run(args, i, generator, target, transform, save_dir_img, idx_to_class)
        if dist.get_rank() == 0:
            logger.info(f'finish recovering label {i}')
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp', type=int, default=1, help='the number of experiment for logging')

    # model and dataset setting
    parser.add_argument('--target', type=str, default='vgg16')
    parser.add_argument('--dataset', type=str, default='umdfaces')
    parser.add_argument('--resolution', type=int, default=112)
    parser.add_argument('--attack', type=str, default='vanilla')
    parser.add_argument('--gan_epochs', type=int, default=10)

    # optimization setting
    parser.add_argument('--num_classes', type=int, default=200) # 1000)
    parser.add_argument('--num_candidates', type=int, default=200)
    parser.add_argument('--num_final', type=int, default=50)
    parser.add_argument('--trunc_p', type=float, default=0.5)
    parser.add_argument('--trunc_c', type=int, default=8)
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--lr', type=int, default=5e-3)
    parser.add_argument('--betas', type=tuple, default=(0.1,0.1))
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=100)

    args = parser.parse_args()
    main(args)