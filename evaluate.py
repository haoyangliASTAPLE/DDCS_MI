import argparse
import os
import glob
import logging
import lpips # image should be RGB, IMPORTANT: normalized to [-1,1]
import time
import shutil
import math
import numpy as np
from tqdm import tqdm
from lpips import upsample, spatial_average
from copy import deepcopy
import torch.cuda
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, distributed, BatchSampler, Subset
from torchvision import transforms
from torchvision.models import Inception3
from models.target import vgg16_bn, inception_v3
from datasets.load import get_dataset, get_recovery
from util.system import init_dist, init_log
from util.fid import load_patched_inception_v3, extract_features, calc_fid

class Iterator:
    def __init__(self, loaders) -> None:
        self.loaders = loaders

    def __iter__(self):
        self.loaders_iter = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        try:
            batch_data, batch_label = [], []
            for loader_iter in self.loaders_iter:
                data, label, _ = next(loader_iter)
                batch_data.append(data)
                batch_label.append(label)
            return torch.cat(batch_data, dim=0), torch.cat(batch_label, dim=0)
        except StopIteration:
            # If any loader is exhausted, reset the iterators
            raise StopIteration

def feature(model: Inception3, img):
    x = model.Conv2d_1a_3x3(img)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = model.maxpool1(x)
    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = model.maxpool2(x)
    x = model.Mixed_5b(x)
    x = model.Mixed_5c(x)
    x = model.Mixed_5d(x)
    x = model.Mixed_6a(x)
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    x = model.Mixed_6e(x)
    x = model.Mixed_7a(x)
    x = model.Mixed_7b(x)
    x = model.Mixed_7c(x)

    return x.view(x.size(0),-1)

def activation(lpip:lpips.LPIPS, img):
    preprocess = lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True)
    img = preprocess(img)
    img = lpip.scaling_layer(img) if lpip.version=='0.1' else img
    outs = lpip.net.forward(img)
    activs = {}
    for k in range(lpip.L):
        activs[k] = lpips.normalize_tensor(outs[k])
    return activs

def distance(lpip:lpips.LPIPS, actv1, actv2, retPerLayer=False):
    diffs = {}
    for kk in range(lpip.L):
        diffs[kk] = (actv1[kk] - actv2[kk]) ** 2
    if(lpip.lpips):
        if(lpip.spatial):
            res = [upsample(lpip.lins[kk](diffs[kk]), out_HW=[256,256]) for kk in range(lpip.L)]
        else:
            res = [spatial_average(lpip.lins[kk](diffs[kk]), keepdim=True) for kk in range(lpip.L)]
    else:
        if(lpip.spatial):
            res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=[256,256]) for kk in range(lpip.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(lpip.L)]

    val = 0
    for l in range(lpip.L):
        val += res[l]

    if(retPerLayer):
        return (val, res)
    else:
        return val

def get_loader(args, load_dir_img):
    train, _ = get_dataset(
        subset='target',
        name=args.dataset,
        crop=args.crop,
        resolution=args.resolution,
        flip=False,
        normalize=True,
    )
    # selected_classes = range(args.num_classes)
    # train = Subset(train, [i for i in range(len(train)) if train[i][1] in selected_classes])
    if args.dataset == 'umdfaces':
        length_200 = 9209
        train = Subset(train, list(range(length_200)))
    target_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=False,
    )

    attack_loaders = []
    for loader_name in args.loaders:
        load_dir_atk = load_dir_img + f'/{loader_name}_recover'
        # load reconstructed datasets
        recovery = get_recovery(
            load_dir_atk,
            normalize=True,
        )
        # recovery = Subset(recovery, [i for i in range(len(recovery)) if recovery[i][1] in selected_classes])
        attack_loaders.append(DataLoader(
            recovery,
            batch_size=args.batch_size // len(args.loaders),
            shuffle=False,
        ))
    attack_iterator = Iterator(attack_loaders)

    return target_loader, attack_iterator, len(train)

def load_dict(sd_actvs_tgt, key):
    load_dir = f'{sd_actvs_tgt}/actvs_{key}.pth.tar'
    actvs_dict:dict = torch.load(load_dir)
    for name, data in actvs_dict.items():
        actvs_dict[name] = data.cuda()
    return actvs_dict

def target_activs(args, lpip:lpips.LPIPS, target_loader, sd_actvs_tgt):
    return_flag = torch.tensor(0).cuda()
    if dist.get_rank() == 0:
        if os.path.exists(sd_actvs_tgt):
            return_flag = torch.tensor(1).cuda()
        else:
            print('make target activations')
            os.makedirs(sd_actvs_tgt)
    dist.barrier()
    dist.broadcast(return_flag, src=0)
    if return_flag:
        return

    pbar = target_loader
    if dist.get_rank() == 0:
        last_key = 0
        actvs_dict = {k:[] for k in range(lpip.L)}
        pbar = tqdm(pbar)
    for data, label in pbar:
        data, label = data.cuda(), label.cuda()
        # distribute
        dist_len = math.ceil(data.shape[0] / args.world_size)
        data_dist = data[args.gpu * dist_len : (args.gpu + 1) * dist_len]

        # get activations
        with torch.no_grad():
            activations = activation(lpip, data_dist)

        # merge activations
        for name, actv in activations.items():
            if not data.shape[0] % args.world_size == 0 and not actv.shape[0] == dist_len:
                zeros = torch.zeros((dist_len - actv.shape[0], ) + actv.shape[1:]).cuda()
                actv = torch.cat((actv, zeros), dim=0)
            temp = [torch.zeros_like(actv) for _ in range(args.world_size)]
            dist.all_gather(temp, actv)
            activations[name] = torch.cat(temp, dim=0)

        # save class-wise actvs
        if dist.get_rank() == 0:
            for i in range(label.shape[0]):
                key = int(label[i])
                # save the result for one target label
                if not key == last_key:
                    # merge
                    save_dict = {}
                    for name, actv in actvs_dict.items():
                        save_dict[name] = torch.cat(actv, dim=0).detach().cpu()
                    # save
                    torch.save(save_dict, f'{sd_actvs_tgt}/actvs_{last_key}.pth.tar')
                    # reset
                    actvs_dict = {k:[] for k in range(lpip.L)}
                # update activs_dict
                for name in actvs_dict.keys():
                    actvs_dict[name].append(activations[name][i].unsqueeze(0))
                last_key = key
        dist.barrier()

    # save the final class's activs
    if dist.get_rank() == 0:
        save_dict = {}
        for name, actv in actvs_dict.items():
            save_dict[name] = torch.cat(actv, dim=0).detach().cpu()
        # save
        torch.save(save_dict, f'{sd_actvs_tgt}/actvs_{last_key}.pth.tar')
    dist.barrier()

def get_ddcs(lpip:lpips.LPIPS, attack_iterator, sd_actvs_tgt, target_total):
    '''
        lpip: LPIPS distance metric.
        attack_iterator: DataLoader for reconstructed dataset.
        sd_actvs_tgt: save directory for activations of target dataset.
        target_total: length of total target dataset samples.
    '''
    # init
    last_key = 0
    actvs_dict = load_dict(sd_actvs_tgt, last_key)
    recovered = torch.full((actvs_dict[0].shape[0],), float('inf'))
    acc_dist, acc_count = torch.zeros(actvs_dict[0].shape[0]), torch.zeros(actvs_dict[0].shape[0])
    ddcs_best, ddcs_avg = 0, 0

    for data, label in tqdm(attack_iterator):
        data, label = data.cuda(), label.cuda()

        with torch.no_grad():
            for i in range(data.shape[0]):
                activations = activation(lpip, data[i].unsqueeze(0))

                # get new target activation for the next label
                key = int(label[i])
                if not key == last_key:
                    # record ddcs best
                    ddcs_best += torch.sum(torch.reciprocal(recovered)).item()
                    del actvs_dict
                    actvs_dict = load_dict(sd_actvs_tgt, key)
                    recovered = torch.full((actvs_dict[0].shape[0],), float('inf'))

                    # record ddcs average
                    for j in range(acc_count.shape[0]):
                        if acc_count[j] > 0:
                            ddcs_avg += 1 / (acc_dist[j] / acc_count[j])
                    acc_dist = torch.zeros(actvs_dict[0].shape[0])
                    acc_count = torch.zeros(actvs_dict[0].shape[0])

                # lpips distance for one atk img to all the yk tgt imgs.
                distances = distance(lpip, activations, actvs_dict).squeeze().cpu() # [bs,1,1,1] -> [bs]

                # whether the tgt img is better recovered
                j = int(torch.argmin(distances))
                recovered[j] = distances[j] if distances[j] < recovered[j] else recovered[j]
                # accumulate
                acc_dist[j] += distances[j]
                acc_count[j] += 1

                last_key = key

    # record ddcs best
    ddcs_best += torch.sum(torch.reciprocal(recovered)).item()
    # record ddcs average
    for j in range(acc_count.shape[0]):
        if acc_count[j] > 0:
            ddcs_avg += 1 / (acc_dist[j] / acc_count[j])

    return ddcs_best / target_total, ddcs_avg / target_total

def load_evaluator(args):
    nc = 1000
    if args.dataset == 'stanforddogs':
        nc = 120
    # load evaluator
    load_dir_evaluator = f'./.saved_models/seed_0/{args.dataset}/inceptionv3'
    evaluator = inception_v3(num_classes=nc, aux_logits=True, init_weights=True).cuda()
    loaded_param = torch.load(f'{load_dir_evaluator}/target.pth')
    evaluator.load_state_dict(loaded_param)
    evaluator.eval()
    return evaluator

def target_feat(args, target_loader):
    feat_dir = f'./.saved_models/{args.dataset}_feat'
    return_flag = torch.tensor(0).cuda()
    if dist.get_rank() == 0:
        if os.path.exists(feat_dir):
            return_flag = torch.tensor(1).cuda()
        else:
            print('make target features')
            os.makedirs(feat_dir)
    dist.barrier()
    dist.broadcast(return_flag, src=0)
    if return_flag:
        return

    # load evaluator
    evaluator = load_evaluator(args)
    resize = transforms.Resize((299))

    # init features dict
    feat_dict = {}
    for i in range(args.num_classes):
        feat_dict[i] = []

    # record target distance
    pbar = target_loader
    if dist.get_rank() == 0:
        pbar = tqdm(pbar)
    for data, label in pbar:
        data, label = data.cuda(), label.cuda()
        # distribute
        dist_len = math.ceil(data.shape[0] / args.world_size)
        data_dist = data[args.gpu * dist_len : (args.gpu + 1) * dist_len]

        # target features
        with torch.no_grad():
            features = feature(evaluator, resize(data_dist))

        # merge features
        if not data.shape[0] % args.world_size == 0 and not features.shape[0] == dist_len:
            zeros = torch.zeros((dist_len - features.shape[0], ) + features.shape[1:]).cuda()
            features = torch.cat((features, zeros), dim=0)
        temp = [torch.zeros_like(features) for _ in range(args.world_size)]
        dist.all_gather(temp, features)
        features = torch.cat(temp, dim=0)

        # save features into the dict
        if dist.get_rank() == 0:
            for i in range(label.shape[0]):
                key = int(label[i])
                feat_dict[key].append(features[i].unsqueeze(0))
        dist.barrier()

    if dist.get_rank() == 0:
        # postprocess, save and return
        for name, data in feat_dict.items():
            save_param = torch.cat(data, dim=0).detach().cpu()
            torch.save(save_param, f'{feat_dir}/feat_{name}.pth')

def get_acc_n_dist(args, attack_iterator):
    # load evaluator
    evaluator = load_evaluator(args)
    resize = transforms.Resize((299))

    # load target features
    feat_dir = f'./.saved_models/{args.dataset}_feat'

    # get recovery features
    pdist = nn.PairwiseDistance(p=2)
    knn_dist, correct_top1, correct_top5, total, last_key = 0, 0, 0, 0, -1
    for data, label in attack_iterator:
        data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            features = feature(evaluator, resize(data))
            logits = evaluator(resize(data))
            outputs = torch.softmax(logits, axis=1)
            _, predicted = outputs.topk(5, 1)

        # calculate statistics
        correct_top1 += predicted[:, 0].eq(label).sum().item()
        correct_top5 += predicted.eq(label.view(-1, 1)).sum().item()
        total += label.shape[0]
        for i in range(label.shape[0]):
            key = int(label[i])
            if not key == last_key:
                target_features = torch.load(f'{feat_dir}/feat_{key}.pth').cuda()
            distances = pdist(features[i], target_features)
            knn_dist += torch.min(distances).item() ** 2
            last_key = key
    return correct_top1 / total, correct_top5 / total, knn_dist / total

def get_fid(args, target_loader, attack_iterator):
    # load inception
    inception = load_patched_inception_v3().cuda()
    inception.eval()
    resize = transforms.Resize((256))

    # get target features
    target_fid_dir = f'./.saved_models/{args.dataset}_fid.npz'
    if os.path.exists(target_fid_dir):
        loaded_param = np.load(target_fid_dir)
        target_mean = loaded_param['mean']
        target_cov = loaded_param['cov']
    else:
        feature_list = []
        for data, label in tqdm(target_loader):
            data, label = data.cuda(), label.cuda()
            features = inception(resize(data))[0].view(data.shape[0], -1)
            feature_list.append(features)
        features = torch.cat(feature_list, dim=0).cpu().numpy()

        # get statistics and save
        target_mean = np.mean(features, 0)
        target_cov = np.cov(features, rowvar=False)
        np.savez(target_fid_dir,
                mean=target_mean,
                cov=target_cov,
                features=features)

    # get recovery features
    feature_list = []
    for data, _ in tqdm(attack_iterator):
        data = data.cuda()
        features = inception(resize(data))[0].view(data.shape[0], -1)
        feature_list.append(features)
    features = torch.cat(feature_list, dim=0).cpu().numpy()

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)
    fid = calc_fid(sample_mean, sample_cov, target_mean, target_cov)
    return fid

def compute_embedding(args, inception:nn.Module, dataloader):
    inception.eval()
    resize = transforms.Resize((256))

    # initialize
    pred_dict = {}
    for i in range(args.num_classes):
        pred_dict[i] = []

    for data, label in dataloader:
        data = data.cuda()
        with torch.no_grad():
            features = inception(resize(data))[0].view(data.shape[0], -1)

        for i in range(label.shape[0]):
            key = int(label[i])
            pred_dict[key].append(features[i].unsqueeze(0))

    # collapse the dict
    for name, data in pred_dict.items():
        pred_dict[name] = torch.cat(data, dim=0)

    return pred_dict

def prcd(args, target_loader, attack_loader, k=3):
    # load inception
    inception = load_patched_inception_v3().cuda()
    inception.eval()

    precision_list, recall_list, density_list, coverage_list = [], [], [], []
    embedding_dict_real = compute_embedding(args, inception, target_loader)
    embedding_dict_fake = compute_embedding(args, inception, attack_loader)
    for i in range(args.num_classes):
        with torch.no_grad():
            embedding_real = embedding_dict_real[i]
            embedding_fake = embedding_dict_fake[i]

            pair_dist_real = torch.cdist(embedding_real, embedding_real, p=2)
            pair_dist_real = torch.sort(pair_dist_real, dim=1, descending=False)[0]
            pair_dist_fake = torch.cdist(embedding_fake, embedding_fake, p=2)
            pair_dist_fake = torch.sort(pair_dist_fake, dim=1, descending=False)[0]
            radius_real = pair_dist_real[:, k]
            radius_fake = pair_dist_fake[:, k]

            # Compute precision
            distances_fake_to_real = torch.cdist(embedding_fake, embedding_real, p=2)
            min_dist_fake_to_real, nn_real = distances_fake_to_real.min(dim=1)
            precision = (min_dist_fake_to_real <= radius_real[nn_real]).float().mean()
            precision_list.append(precision.cpu().item())

            # Compute recall
            distances_real_to_fake = torch.cdist(embedding_real, embedding_fake, p=2)
            min_dist_real_to_fake, nn_fake = distances_real_to_fake.min(dim=1)
            recall = (min_dist_real_to_fake <= radius_fake[nn_fake]).float().mean()
            recall_list.append(recall.cpu().item())

            # Compute density
            num_samples = distances_fake_to_real.shape[0]
            sphere_counter = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0).mean()
            density = sphere_counter / k
            density_list.append(density.cpu().item())

            # Compute coverage
            num_neighbors = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0)
            coverage = (num_neighbors > 0).float().mean()
            coverage_list.append(coverage.cpu().item())
    # Compute mean over targets
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    density = np.mean(density_list)
    coverage = np.mean(coverage_list)
    return precision, recall, density, coverage

def main(args):
    # create save dir
    load_dir_img = f'./.saved_figs/seed_{args.seed}/{args.dataset}/{args.target}'
    sd_actvs_tgt = f'./.saved_models/{args.dataset}_actvs'
    # init logger
    log_dir='./.logger'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{log_dir}/exp{args.exp}_seed{args.seed}.txt', mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # get loader
    target_loader, attack_iterator, target_total = get_loader(args, load_dir_img)

    # load lpips
    lpip = lpips.LPIPS(net='vgg').cuda()

    # record evaluator distance and activations
    target_activs(args, lpip, target_loader, sd_actvs_tgt)
    target_feat(args, target_loader)

    # get results
    precision, recall, density, coverage = 0, 0, 0, 0
    acc_top1, acc_top5, distances, fid, ddcs_best, ddcs_avg = 0, 0, 0, 0, 0, 0
    fid = get_fid(args, target_loader, attack_iterator)
    acc_top1, acc_top5, distances = get_acc_n_dist(args, attack_iterator)
    ddcs_best, ddcs_avg = get_ddcs(lpip, attack_iterator, sd_actvs_tgt, target_total)
    precision, recall, density, coverage = prcd(args, target_loader, attack_iterator)

    report = []
    report.append(f'{args.dataset}-{args.target}-{args.loaders}-seed{args.seed}')
    report.append('Accuracy-Top1: {:.4f}%, Accuracy-Top5: {:.4f}%;'.format(acc_top1*100, acc_top5*100))
    report.append('KNN distance {:.4f};'.format(distances))
    report.append('FID {:.4f};'.format(fid))
    report.append('DDCS (Best) {:.4f}; DDCS (Average) {:.4f};'.format(ddcs_best, ddcs_avg))
    report.append('Precision {:.4f}; Recall {:.4f}; '.format(precision, recall) + \
                  'Density {:.4f}; Coverage {:.4f};'.format(density, coverage))
    for line in report:
        logger.info(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp', type=int, default=1, help='the number of experiment for logging')

    # model and dataset setting
    parser.add_argument('--num_classes', type=int, default=200) # 1000)
    parser.add_argument('--target', type=str, default='vgg16')
    parser.add_argument('--dataset', type=str, default='umdfaces')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--resolution', type=int, default=112)
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--loaders', nargs='+', type=str, default=['vanilla_10'])

    args = parser.parse_args()
    main(args)