"""
Provide menu or control the flow
"""
import warnings

warnings.filterwarnings("ignore")

from utils import load_config, load_data
import numpy as np
import logging
import os
import random
import torch
import torch_geometric
import torchvision
from train import Trainer
import argparse
import time


def set_seed(seed):
    # Set random seed to ensure reproducibility
    # Ensure predictable random number generation when using CNN-related functions
    torch.backends.cudnn.deterministic = True
    # Disable automatic benchmarking for faster execution, but non-reproducible results
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():  # If CUDA is available
        torch.cuda.manual_seed(seed)  # Set CUDA random seed
        torch.cuda.manual_seed_all(seed)  # Set random seed for all CUDA devices
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python hash seed
    np.random.seed(seed)  # Set NumPy random seed
    torch.manual_seed(seed)  # Set Torch random seed
    random.seed(seed)  # Set Python built-in random seed


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='bio-benchmark')

    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="config/regression/BatmanNet.yaml")

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    return parser.parse_known_args()[0]


if __name__ == '__main__':

    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = load_config(args.config)

    set_seed(args.seed)
    start_time = time.strftime("%Y%m%d-%H%M%S")
    exp_name = cfg.task.model['class'] + f"-{start_time}"

    perf_save_path = os.path.join(cfg['output_dir'], cfg.dataset['class'], exp_name)
    if not os.path.exists(perf_save_path):
        os.makedirs(perf_save_path)

    logger = get_logger(perf_save_path + '/exp.log')
    logger.info(cfg)

    trainer = Trainer(cfg, logger, scheduler=None, model_path=perf_save_path, seed=args.seed)

    if cfg.task['train'] == 'kfold':
        trainer.K_fold_train()
    elif cfg.task['train'] == 'train_test':
        trainer.train_test()
    elif cfg.task['train'] == 'memory_test':
        trainer.mem_speed_bench()
