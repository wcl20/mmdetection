import argparse
import torch
import mmcv
import os.path as osp
import time
from mmcv import Config
from mmdet.apis import init_random_seed
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.utils import collect_env
from mmdet.utils import get_root_logger
from mmdet.utils import setup_multi_processes
from mmdet.utils import update_data_root
from pprint import pprint

def main():

    # Arguments
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    # Config
    cfg = Config.fromfile(args.config)
    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    # System setup
    setup_multi_processes(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # GPU Device ID
    cfg.gpu_ids = [args.gpu_id]
    # Distributed Training
    distributed = False
    # Deterministic options for CUDNN backend
    deterministic = False

    # Working directory
    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # Logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Environment Information
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])

    # Seed
    seed = init_random_seed(args.seed)
    set_random_seed(seed, deterministic=deterministic)
    cfg.seed = seed

    # Meta data
    meta = dict()
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(args.config)

    logger.info("Environment Info:")
    logger.info("-" * 60)
    logger.info(env_info)
    logger.info("-" * 60)
    logger.info(f"Distributed Training: {distributed}")
    logger.info("Config:")
    logger.info(cfg.pretty_text)
    logger.info(f"Deterministic: {deterministic}")

    datasets = [build_dataset(cfg.data.train)]
    item = next(iter(datasets[0]))
    pprint(item)






if __name__ == '__main__':
    main()
