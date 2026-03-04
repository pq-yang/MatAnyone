import os
from os import path
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from matanyone.dataset.static_dataset import SyntheticVideoDataset
from matanyone.dataset.vos_dataset import VOSTrainDataset
from matanyone.dataset.vm_dataset import VideoMatteDataset
from matanyone.dataset.im_dataset import ImageMatteDataset


_local_rank = int(os.environ['LOCAL_RANK'])
log = logging.getLogger()


# Re-seed randomness every time we start a worker
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + _local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    log.debug(f'Worker {worker_id} re-seeded with seed {worker_seed} in rank {_local_rank}')


def setup_seg_img_training_datasets(cfg, stage_cfg):
    root = cfg.data.image_datasets.base
    datasets = ["COCO", "SPD"]
    dataset_configs = [cfg.data.image_datasets[d] for d in datasets]
    dataset_tuples = []
    for d_cfg in dataset_configs:
        try:
            dataset_tuples.append((path.join(root, d_cfg.imgdir), path.join(root, d_cfg.anndir), path.join(root, d_cfg.annfile), d_cfg.multiplier)) 
        except:
            dataset_tuples.append((path.join(root, d_cfg.imgdir), path.join(root, d_cfg.anndir), None, d_cfg.multiplier)) 

    dataset = SyntheticVideoDataset(dataset_tuples,
                                    seq_length=stage_cfg.seq_length,
                                    max_num_obj=stage_cfg.num_objects,
                                    size=stage_cfg.crop_size[0])

    batch_size = stage_cfg.batch_size
    num_workers = cfg.num_workers
    sampler, loader = construct_loader(dataset, batch_size, num_workers, _local_rank)

    return dataset, sampler, loader


def setup_seg_video_training_datasets(cfg, stage_cfg, max_skip):
    root = cfg.data.vos_datasets.base
    datasets = ['YouTubeVOS']
    dataset_configs = [cfg.data.vos_datasets[d] for d in datasets]

    dataset_configs = {
        name: {
            'im_root': path.join(root, d_cfg.image_directory),
            'gt_root': path.join(root, d_cfg.mask_directory),
            'max_skip': max_skip // d_cfg.frame_interval,
            'multiplier': d_cfg.multiplier,
        }
        for name, d_cfg in zip(datasets, dataset_configs)
    }

    dataset = VOSTrainDataset(dataset_configs,
                                   seq_length=stage_cfg.seq_length,
                                   max_num_obj=stage_cfg.num_objects,
                                   size=stage_cfg.crop_size[0], 
                                   merge_probability=stage_cfg.merge_probability)

    batch_size = stage_cfg.batch_size
    num_workers = cfg.num_workers
    sampler, loader = construct_loader(dataset, batch_size, num_workers, _local_rank)

    log.info(f'Using a max skip of {max_skip} frames')

    return dataset, sampler, loader


def setup_matting_img_training_datasets(cfg, stage_cfg):
    root = cfg.data.im_datasets.base
    datasets = ['ImageMatte'] 
    
    dataset_configs = [cfg.data.im_datasets[d] for d in datasets]

    dataset_configs = {
        name: {
            'fgr_img_directory': path.join(root, d_cfg.fgr_img_directory),
            'bg_img_directory': path.join(root, d_cfg.bg_img_directory),
            'bg_video_directory': path.join(root, d_cfg.bg_video_directory),
            'multiplier': d_cfg.multiplier,
        }
        for name, d_cfg in zip(datasets, dataset_configs)
    }

    dataset = ImageMatteDataset(dataset_configs,
                                seq_length=stage_cfg.seq_length,
                                max_num_obj=stage_cfg.num_objects,
                                size=stage_cfg.crop_size[0], 
                                merge_probability=stage_cfg.merge_probability)

    batch_size = stage_cfg.batch_size
    num_workers = cfg.num_workers
    sampler, loader = construct_loader(dataset, batch_size, num_workers, _local_rank)

    return dataset, sampler, loader


def setup_matting_video_training_datasets(cfg, stage_cfg, max_skip):
    root = cfg.data.vm_datasets.base
    datasets = ['VM800'] # 'VideoMatte240K' as alternative
    dataset_configs = [cfg.data.vm_datasets[d] for d in datasets]

    dataset_configs = {
        name: {
            'fgr_video_directory': path.join(root, d_cfg.fgr_video_directory),
            'bg_img_directory': path.join(root, d_cfg.bg_img_directory),
            'bg_video_directory': path.join(root, d_cfg.bg_video_directory),
            'max_skip': max_skip // d_cfg.frame_interval,
            'multiplier': d_cfg.multiplier,
        }
        for name, d_cfg in zip(datasets, dataset_configs)
    }

    dataset = VideoMatteDataset(dataset_configs,
                                   seq_length=stage_cfg.seq_length,
                                   max_num_obj=stage_cfg.num_objects,
                                   size=stage_cfg.crop_size[0], 
                                   merge_probability=stage_cfg.merge_probability)

    batch_size = stage_cfg.batch_size
    num_workers = cfg.num_workers
    sampler, loader = construct_loader(dataset, batch_size, num_workers, _local_rank)

    log.info(f'Using a max skip of {max_skip} frames')

    return dataset, sampler, loader


def construct_loader(dataset, batch_size, num_workers, local_rank):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    rank=local_rank,
                                                                    shuffle=True)
    train_loader = DataLoader(dataset,
                              batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              worker_init_fn=worker_init_fn,
                              drop_last=True,
                              persistent_workers=True)
    return train_sampler, train_loader
