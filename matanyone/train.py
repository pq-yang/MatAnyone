import os
import math

import logging
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig

import random
import numpy as np
import torch
import torch.distributed as distributed

from matanyone.model.trainer import Trainer
from matanyone.dataset.setup_training_data import setup_seg_img_training_datasets, setup_seg_video_training_datasets, \
setup_matting_img_training_datasets, setup_matting_video_training_datasets
from matanyone.utils.logger import TensorboardLogger

local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend="nccl")
    local_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


def info_if_rank_zero(msg):
    if local_rank == 0:
        log.info(msg)


@hydra.main(version_base='1.3.2', config_path='config', config_name='train_config.yaml')
def train(cfg: DictConfig):
    # initial setup
    distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir
    info_if_rank_zero(f'All configuration: {cfg}')
    info_if_rank_zero(f'Number of detected GPUs: {num_gpus}')

    # cuda setup
    torch.cuda.set_device(local_rank)
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # number of dataloader workers
    cfg.num_workers //= num_gpus
    info_if_rank_zero(f'Number of dataloader workers (per GPU): {cfg.num_workers}')

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(run_dir, logging.getLogger(), enabled_tb=(local_rank == 0))

    # training stages
    stages = []
    if cfg.stage_1.enabled:
        stages.append('stage_1')
    if cfg.stage_2.enabled:
        stages.append('stage_2')
    if cfg.stage_3.enabled:
        stages.append('stage_3')
    info_if_rank_zero(f'Enabled stages: {stages}')

    weights_in_memory = None  # for transferring weights between stages
    for stage in stages:
        # Set seeds to ensure the same initialization
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # setting up configurations
        stage_cfg = cfg[stage]
        info_if_rank_zero(f'Training stage: {stage}')
        info_if_rank_zero(f'Training configuration: {stage_cfg}')
        stage_cfg.batch_size //= num_gpus
        info_if_rank_zero(f'Batch size (per GPU): {stage_cfg.batch_size}')

        # construct the trainer
        trainer = Trainer(cfg, stage_cfg, log=log, run_path=run_dir).train()

        # load previous checkpoint if needed
        if cfg['checkpoint'] is not None:
            curr_iter = trainer.load_checkpoint(cfg['checkpoint'])
            cfg['checkpoint'] = None
            info_if_rank_zero('Model checkpoint loaded!')
        else:
            curr_iter = 0

        # load previous network weights if needed
        if weights_in_memory is not None:
            info_if_rank_zero('Loading weights from the previous stage')
            trainer.load_weights_in_memory(weights_in_memory)
            weights_in_memory = None
        elif cfg['weights'] is not None:
            info_if_rank_zero('Loading weights from the disk')
            trainer.load_weights(cfg['weights'])
            cfg['weights'] = None

        # determine time to change max skip
        total_iterations = stage_cfg['num_iterations']
        if 'max_skip_schedule' in stage_cfg:
            max_skip_schedule = stage_cfg['max_skip_schedule']
            increase_skip_fraction = stage_cfg['max_skip_schedule_fraction']
            change_skip_iter = [round(total_iterations * f) for f in increase_skip_fraction]
            # Skip will only change after an epoch, not in the middle
            log.info(f'The skip value will change at these iters: {change_skip_iter}')
        else:
            change_skip_iter = []
        change_skip_iter.append(total_iterations + 1)  # dummy value to avoid out of index error

        # setup datasets
        print(f"========= Setup {stage} Datasets =========")
        seg_video_dataset, seg_video_sampler, seg_video_loader = setup_seg_video_training_datasets(cfg, stage_cfg, max_skip_schedule[0])
        seg_img_dataset, seg_img_sampler, seg_img_loader = setup_seg_img_training_datasets(cfg, stage_cfg)
        
        if stage_cfg["use_video"]:
            dataset, sampler, loader = setup_matting_video_training_datasets(cfg, stage_cfg, max_skip_schedule[0])
        else:
            dataset, sampler, loader = setup_matting_img_training_datasets(cfg, stage_cfg)
        
        log.info(f'Number of matting training samples: {len(dataset)}')
        log.info(f'Number of matting training batches: {len(loader)}')

        log.info(f'Number of seg video training samples: {len(seg_video_dataset)}')
        log.info(f'Number of seg video training batches: {len(seg_video_loader)}')

        log.info(f'Number of seg img training samples: {len(seg_img_dataset)}')
        log.info(f'Number of seg img training batches: {len(seg_img_loader)}')

        # determine max epoch
        total_epoch = math.ceil(total_iterations / len(loader))
        current_epoch = curr_iter // len(loader)
        log.info(f'We will approximately use {total_epoch} epochs.')

        # training loop
        try:
            # Need this to select random bases in different workers
            np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)
            seg_video_iterator = None
            seg_img_iterator = None
            while curr_iter < total_iterations:
                # Crucial for randomness!
                sampler.set_epoch(current_epoch)
                current_epoch += 1
                log.debug(f'Current epoch: {current_epoch}')

                trainer.train()
                for data in loader:
                    # Update skip if needed
                    if curr_iter >= change_skip_iter[0] and \
                        (stage_cfg.name != 'stage_1' and stage_cfg.use_video):
                        while curr_iter >= change_skip_iter[0]:
                            cur_skip = max_skip_schedule[0]
                            max_skip_schedule = max_skip_schedule[1:]
                            change_skip_iter = change_skip_iter[1:]
                        log.info(f'Changing max skip to {cur_skip=}')
                        _, sampler, loader = setup_matting_video_training_datasets(cfg, stage_cfg, cur_skip)
                        break
                    
                    # Matting Pass
                    trainer.do_pass(data, curr_iter, img_pass=(not stage_cfg.use_video), clamp_mat=(curr_iter >= stage_cfg.clamp_start))

                    # Segmentation Pass
                    if curr_iter % 2 == 0:
                        try:
                            seg_data = next(seg_video_iterator)
                        except:
                            seg_video_sampler.set_epoch(seg_video_sampler.epoch + 1)
                            seg_video_iterator = iter(seg_video_loader)
                            seg_data = next(seg_video_iterator)

                        trainer.do_pass(seg_data, curr_iter, seg_pass=True, img_pass=False)
                    else:
                        try:
                            seg_data = next(seg_img_iterator)
                        except:
                            seg_img_sampler.set_epoch(seg_img_sampler.epoch + 1)
                            seg_img_iterator = iter(seg_img_loader)
                            seg_data = next(seg_img_iterator)
                        
                        trainer.do_pass(seg_data, curr_iter, seg_pass=True, img_pass=True)

                    # Core Supervision Pass (seg_mat=True)
                    if stage_cfg['core_supervision']:
                        trainer.do_pass(seg_data, curr_iter, seg_pass=True, seg_mat=True)

                    curr_iter += 1

                    if curr_iter >= total_iterations:
                        break
        finally:
            if not cfg.debug:
                trainer.save_weights(curr_iter)
                trainer.save_checkpoint(curr_iter)

        torch.cuda.empty_cache()
        weights_in_memory = trainer.weights()

    # clean-up
    distributed.destroy_process_group()


if __name__ == '__main__':
    train()
