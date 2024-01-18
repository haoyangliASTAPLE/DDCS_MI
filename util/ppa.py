import numpy as np
import math
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms

def create_target_vector(target_classes='all', num_candidates=5): # 200):
    targets = None
    if type(target_classes) is list:
        targets = torch.tensor(target_classes)
        targets = torch.repeat_interleave(targets, num_candidates)
    elif target_classes == 'all':
        targets = torch.tensor([i for i in range(1000)])
        targets = torch.repeat_interleave(targets, num_candidates)
    elif type(target_classes) == int:
        targets = torch.full(size=(num_candidates, ),
                                 fill_value=target_classes)
    else:
        raise Exception(
            f' Please specify a target class or state a target vector.')

    return targets.cuda()

def find_initial_w(args,
                   generator,
                   target_model,
                   targets,
                   transform,
                   search_space_size=5000,
                   horizontal_flip=False,
                   filepath=None,
                   truncation_psi=0.7,
                   truncation_cutoff=18):
    """Find good initial starting points in the style space.

    Args:
        generator (torch.nn.Module): StyleGAN2 model
        target_model (torch.nn.Module): [description]
        target_cls (int): index of target class.
        search_space_size (int): number of potential style vectors.
        clip (boolean, optional): clip images to [-1, 1]. Defaults to True.
        center_crop (int, optional): size of the center crop. Defaults 768.
        resize (int, optional): size for the resizing operation. Defaults to 224.
        horizontal_flip (boolean, optional): apply horizontal flipping to images. Defaults to true.
        filepath (str): filepath to save candidates.
        truncation_psi (float, optional): truncation factor. Defaults to 0.7.
        truncation_cutoff (int, optional): truncation cutoff. Defaults to 18.
        batch_size (int, optional): batch size. Defaults to 25.

    Returns:
        torch.tensor: style vectors with highest confidences on the target model and target class.
    """

    z = torch.from_numpy(
        np.random.RandomState(args.seed).randn(search_space_size,
                                        generator.z_dim)).cuda() # [5000, 512]
    dist.broadcast(z, src=0)
    c = None
    target_model.eval()

    with torch.no_grad():
        confidences = []
        final_candidates = []
        # final_confidences = []
        candidates = generator.mapping(
            z, c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff
        )
        candidate_dataset = TensorDataset(candidates) # [5000, 14, 512]
        pbar = DataLoader(candidate_dataset, batch_size=args.batch_size)
        if dist.get_rank() == 0:
            pbar = tqdm(pbar, desc='Find initial style vector w')
        for w in pbar:
            # split data and distribute all GPUs
            w = w[0] # [w] -> w, [bs,14,512]
            w_split = torch.split(w, w.shape[0] // args.world_size)
            w_dist = w_split[args.gpu] # [bs//num_gpus,14,512] [25,14,512]

            imgs = generator.synthesis(w_dist,
                                       noise_mode='const',
                                       force_fp32=True) # [25,3,256,256]
            # Adjust images and perform augmentation
            imgs = [imgs]
            if horizontal_flip:
                imgs.append(F.hflip(imgs[0]))

            target_conf = None
            for im in imgs: # if flip then len(imgs) == 2 otherwise 1
                if target_conf is not None:
                    target_conf += target_model(transform(im)).softmax(dim=1) / len(imgs)
                else:
                    target_conf = target_model(transform(im)).softmax(dim=1) / len(imgs)

            # merge results from all GPUs
            merge = [torch.zeros_like(target_conf) for _ in range(args.world_size)]
            dist.all_gather(merge, target_conf)
            merge = torch.cat(merge, dim=0)

            confidences.append(merge)
        confidences = torch.cat(confidences, dim=0) # [5000,1000] [search_space_size,num_classes]

        # split data and distribute all GPUs
        dist_len = len(targets) // args.world_size
        targets_dist = targets[args.gpu * dist_len : (args.gpu + 1) * dist_len]
        final_candidates_dist = []
        for target in targets_dist:
            sorted_conf, sorted_idx = confidences[:, target].sort(descending=True)
            final_candidates_dist.append(candidates[sorted_idx[0]].unsqueeze(0))
            # final_confidences.append(sorted_conf[0].cpu().item())
            # Avoid identical candidates for the same target
            confidences[sorted_idx[0], target] = -1.0
        final_candidates_dist = torch.cat(final_candidates_dist, dim=0).cuda() # [1250,14,512]
        # final_confidences = [np.round(c, 2) for c in final_confidences]

        # merge results from all GPUs
        final_candidates = [torch.zeros_like(final_candidates_dist) for _ in range(args.world_size)]
        dist.all_gather(final_candidates, final_candidates_dist)
        final_candidates = torch.cat(final_candidates, dim=0).cuda()

        if dist.get_rank() == 0:
            print(f'Found {final_candidates.shape[0]} initial style vectors.')
            if filepath:
                torch.save(final_candidates, filepath)
                print(f'Candidates have been saved to {filepath}')

    return final_candidates.cuda()

def poincare_loss(outputs, targets, xi=1e-4):
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(1)
    # Create one-hot encoded target vector
    v = torch.clip(torch.eye(outputs.shape[-1])[targets] - xi, 0, 1)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1)**2
    v_norm_squared = torch.norm(v, p=2, dim=1)**2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1)**2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) *
                                     (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss.mean()

def scores_by_transform(imgs,
                        targets,
                        target_model,
                        transforms,
                        iterations=100):

    score = torch.zeros_like(
        targets, dtype=torch.float32).to(imgs.device)

    with torch.no_grad():
        for i in range(iterations):
            imgs_transformed = transforms(imgs)
            prediction_vector = target_model(imgs_transformed).softmax(dim=1)
            score += torch.gather(prediction_vector, 1,
                                  targets.unsqueeze(1)).squeeze()
        score = score / iterations
    return score

def perform_final_selection(
        args,
        batch_size,
        w,
        synthesis,
        targets,
        target_model,
        samples_per_target=50,
        iterations=100,
    ):
    target_values = set(targets.cpu().tolist())
    final_targets = []
    final_w = []
    target_model.eval()

    crop = 112
    if args.dataset == 'stanforddogs':
        crop = 224
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(crop, crop),
                            scale=(0.5, 0.9),
                            ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(0.5)
    ])

    pbar = target_values
    if dist.get_rank() == 0:
        pbar = tqdm(pbar)
    for target in pbar:
        mask = torch.where(targets == target, True, False)
        w_masked = w[mask]
        targets_masked = targets[mask]
        scores = []
        for i in range(math.ceil(w_masked.shape[0] / batch_size)):
            w_batch = w_masked[i * batch_size : (i + 1) * batch_size]
            w_dist = torch.split(w_batch, w_batch.shape[0] // args.world_size)[args.gpu]
            t = targets_masked[i * batch_size : (i + 1) * batch_size]
            t_dist = torch.split(t, t.shape[0] // args.world_size)[args.gpu]
            with torch.no_grad():
                imgs = synthesis(w_dist, noise_mode='const', force_fp32=True)

            score_dist = scores_by_transform(imgs,
                                             t_dist,
                                             target_model,
                                             transform,
                                             iterations)
            # merge
            score = [torch.zeros_like(score_dist) for _ in range(args.world_size)]
            dist.all_gather(score, score_dist)
            scores.append(torch.cat(score, dim=0))
        scores = torch.cat(scores, dim=0).cpu()
        indices = torch.sort(scores, descending=True).indices
        selected_indices = indices[:samples_per_target]
        final_targets.append(targets_masked[selected_indices].cpu())
        final_w.append(w_masked[selected_indices].cpu())

    final_targets = torch.cat(final_targets, dim=0)
    final_w = torch.cat(final_w, dim=0)
    return final_w.detach(), final_targets.detach()