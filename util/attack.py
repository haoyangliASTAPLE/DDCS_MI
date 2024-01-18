from tqdm import tqdm
from copy import deepcopy
import torch.cuda
import torch.backends.cudnn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.nn.parallel
import torch.nn.functional as F
import lpips
import gc
from torch import nn, optim, autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.models import wide_resnet50_2, inception_v3
from models.target import vgg16_bn, resnet50, resnet101, resnet152, alexnet
from models.evolve import IR_50, IR_101, IR_152, IR_SE_101, IR_SE_152, IR_SE_50
from util.ada import *
from util.hvp import GANHVPOperator, lanczos

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        targetprobs = F.softmax(target, dim=1)
        logprobs = F.log_softmax(input, dim=1)
        return  - (targetprobs * logprobs).sum() / input.shape[0]

def hess_vec_prod(z, distance, generator):
    cutoff = 20
    metricHVP = GANHVPOperator(generator, z, distance, preprocess=lambda img: img)
    eigvals, eigvects = lanczos(metricHVP, num_eigenthings=cutoff, use_gpu=True)
    eigvects = eigvects.T  # note the output shape from lanczos is different from that of linalg.eigh, row is eigvec
    return torch.from_numpy(eigvals).cuda(), torch.from_numpy(eigvects).cuda()

def init_attack(args, load_dir_target, generator):
    target, resize, criterion, eigvals, eigvects = \
    None,   None,   None,      None,    None
    if args.attack == 'vanilla':
        return target, resize, criterion, eigvals, eigvects

    # number of classes
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
    loaded_param = torch.load(f'{load_dir_target}/target.pth')
    target.load_state_dict(loaded_param)
    target.eval()

    # create resize
    if args.dataset == 'umdfaces':
        crop = 200
        resolution = 112
    elif args.dataset == 'stanforddogs':
        crop = 256
        resolution = 224
    resize = transforms.Compose([
        transforms.CenterCrop(crop), # 256 -> 200
        transforms.Resize((resolution)), # 200 -> 112
    ])

    # criterion and competitor
    if args.attack == 'hloss' or args.attack == 'our':
        criterion = EntropyLoss()

    # get hvp
    distance = lpips.LPIPS(net='vgg').cuda()
    eigvals, eigvects = [], []
    pbar = range(args.batch_size)
    if dist.get_rank() == 0:
        print('start get hvp')
        pbar = tqdm(pbar)
    for _ in pbar:
        z = torch.rand(1, args.z_dim, device='cuda')
        eigval, eigvect = hess_vec_prod(z, distance, deepcopy(generator))
        eigvals.append(eigval.unsqueeze(0))
        eigvects.append(eigvect.unsqueeze(0))
    eigvals = torch.cat(eigvals, dim=0)
    eigvects = torch.cat(eigvects, dim=0)
    torch.save(eigvects.detach().cpu(),
               f'{load_dir_target}/eigvects_{args.attack}_{args.epochs}_{args.gpu}.pth')
    eigvects = None

    return target, resize, criterion, eigvals, eigvects

def adv_loss(args, fake, fake_aug, resize, target, criterion, eigvals, eigvects, loss):
    if args.attack == 'vanilla':
        return loss

    else:
        lamda_h = 1e0
        # entropy loss
        fake.requires_grad_(True)
        loss_g_adv = lamda_h * criterion(target(resize(fake_aug)))

    # calculate dL/dG(z)
    gc.collect()
    torch.cuda.empty_cache()
    fake_grad = autograd.grad( # [bs,3,256,256]
        loss_g_adv,
        fake,
        retain_graph=True,
    )[0]

    if args.attack == 'our':
        # load eigen vectors
        sl_dir_model = f'./.saved_models/seed_{args.seed}/{args.dataset}/{args.target}'
        eigvects = torch.load(f'{sl_dir_model}/eigvects_{args.attack}_{args.epochs}_{args.gpu}.pth').cuda()

        # get P(dL/dG(z))
        size = fake_grad[0].shape
        for i in range(fake_grad.size(0)):
            vect = torch.diag(torch.reciprocal(eigvals[i])) @ eigvects[i].T @ fake_grad[i].view(-1)
            fake_grad[i] = torch.reshape(eigvects[i] @ vect, size)

    # get G's gradient as P(dL/dG(z)) * dG(z)/dG
    fake_grad_p = (fake * deepcopy(fake_grad)).sum()

    return loss + fake_grad_p
