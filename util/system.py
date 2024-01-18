import logging
import os
import numpy as np
import torch.cuda
import torch.backends.cudnn
import torch.distributed as dist

def init_dist(args):
    # distributed setup
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.gpu)
    args.dist_url = 'env://'
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    np.random.seed(args.seed * args.world_size + args.gpu)
    torch.manual_seed(args.seed * args.world_size + args.gpu)
    torch.backends.cudnn.benchmark = True
    return args

def init_log(args, log_dir=f'./.logger'):
    if dist.get_rank() == 0 and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{log_dir}/exp{args.exp}_seed{args.seed}.txt', mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger