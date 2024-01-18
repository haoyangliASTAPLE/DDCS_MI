import argparse
import os
import sys
import subprocess
import logging
import tempfile
import numpy as np
from tqdm import tqdm
import torch.cuda
from torch import nn, optim
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data import DataLoader, distributed, BatchSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import wide_resnet50_2, inception_v3
from models.target import vgg16_bn, resnet50, resnet101, resnet152, alexnet
from models.evolve import IR_50, IR_101, IR_152, IR_SE_101, IR_SE_152, IR_SE_50
from datasets.load import get_dataset
from util.system import init_dist, init_log

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def schedule_lr(optimizer:optim.SGD):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    if dist.get_rank() == 0:
        print(optimizer)

def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer:optim.SGD):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

def get_loader(args):
    # load dataset
    train, test = get_dataset(
        subset='target',
        name=args.dataset,
        resolution=args.resolution,
        crop=args.crop,
    )
    train_sampler = distributed.DistributedSampler(train)
    train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_loader = DataLoader(train,
                            batch_sampler=train_batch_sampler,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory)
    test_sampler = distributed.DistributedSampler(test, shuffle=False)
    test_loader = DataLoader(test,
                            batch_size=100,
                            sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    if dist.get_rank() == 0:
        print(f'finish loading {args.dataset} dataset')
    return train_loader, test_loader, train_sampler

def init_training(args):
    # number of classes
    nc = 1000
    pretrained = False
    if args.dataset == 'stanforddogs':
        nc = 120
        if not args.arch == 'inceptionv3':
            pretrained = True
    # load model
    if args.arch == 'vgg16':
        model = vgg16_bn(num_classes=nc, pretrained=pretrained).cuda()
    elif args.arch == 'resnet50':
        model = resnet50(num_classes=nc, pretrained=pretrained).cuda()
    elif args.arch == 'irse50':
        model = IR_SE_50(num_classes=nc).cuda()
    elif args.arch == 'alexnet':
        model = alexnet(num_classes=nc, pretrained=pretrained).cuda()
    elif args.arch == 'inceptionv3':
        model = inception_v3(
            num_classes=nc,
            pretrained=pretrained,
            aux_logits=True,
            init_weights=True,
        ).cuda()
    # DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # optimizer
    optimizer = optim.SGD(model.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum, 
                          weight_decay=args.weight_decay)
    if args.loss == 'focal':
        criterion = FocalLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def main(args):
    args = init_dist(args)

    if dist.get_rank() == 0:
        # create save dir
        save_dir_model = f'./.saved_models/seed_{args.seed}/{args.dataset}/{args.arch}'
        if not os.path.exists(save_dir_model):
            os.makedirs(save_dir_model)
        # get logger
        logger = init_log(args)
    dist.barrier()

    # get data loader
    train_loader, test_loader, train_sampler = get_loader(args)
    # initialize training
    model, optimizer, criterion = init_training(args)

    # training
    if dist.get_rank() == 0:
        logger.info(f'---------------------train {args.arch} with {args.dataset}---------------------')

    warm_up_epoch = args.epochs // 25
    warm_up_batch = len(train_loader) * warm_up_epoch
    batch = 1
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        if (epoch - 1) in args.stages:
            schedule_lr(optimizer)

        # train
        model.train()
        pbar = train_loader
        if dist.get_rank() == 0:
            pbar = tqdm(pbar)
        for img, label in pbar:
            if epoch <= warm_up_epoch and batch <= warm_up_batch:
                warm_up_lr(batch, warm_up_batch, args.lr, optimizer)
            img, label = img.cuda(), label.cuda()

            if not args.arch == 'inceptionv3':
                logits = model(img)
                loss = criterion(logits, label).mean()
            else:
                main_logits, aux_logits = model(img)
                main_loss = criterion(main_logits, label)
                aux_loss = criterion(aux_logits, label).sum()
                loss = main_loss + aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch += 1

        # eval
        model.eval()
        correct, total = 0, 0
        if epoch % args.eval_steps == 0 and dist.get_rank() == 0:
            for img, label in tqdm(test_loader):
                img, label = img.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(img)
                outputs = torch.softmax(logits, axis=1)
                _, predicted = torch.max(outputs, dim=1)
                correct += sum(predicted == label)
                total += len(label)
            logger.info(f'epoch {epoch} - accuracy: {correct/total}')

        # save
        if epoch % args.save_steps == 0 and dist.get_rank() == 0:
            save_path = f'{save_dir_model}/target.pth'
            torch.save(model.module.state_dict(), save_path)
            logger.info(f'epoch {epoch} - save model at {save_path}')
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp', type=int, default=1)

    # model and dataset setting
    parser.add_argument('--arch', type=str, default='vgg16')
    parser.add_argument('--dataset', type=str, default='umdfaces')
    parser.add_argument('--resolution', type=int, default=112, help='299 for inceptionv3')
    parser.add_argument('--crop', type=int, default=None)

    # optimization setting
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--loss', type=str, default='focal', help='focal or ce loss')

    # training setting
    parser.add_argument('--epochs', type=int, default=125)
    parser.add_argument('--stages', type=list, default=[35,65,95])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=10, help='how often to save model')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers for data loaders')
    parser.add_argument('--pin_memory', type=int, default=1, help='pin the memory for train loader')

    args = parser.parse_args()
    main(args)
