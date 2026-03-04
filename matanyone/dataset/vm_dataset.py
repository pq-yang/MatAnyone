"""Modified from https://github.com/PeterL1n/RobustVideoMatting/blob/master/dataset/videomatte.py
"""
import os
from os import path
import logging
import random

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from matanyone.dataset.augmentation import MotionAugmentation, TrainFrameSampler

log = logging.getLogger()
local_rank = int(os.environ['LOCAL_RANK'])


class VideoMatteDataset(Dataset):
    
    def __init__(self, data_configs, seq_length=3, max_num_obj=3, size=480, merge_probability=0.0):

        self.configs = data_configs
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size
        self.merge_probability = merge_probability

        self.max_crop_trials = 5  # number of attempts at cropping a single frame
        self.max_seed_trials = 5  # number of attempts at changing the initial seed frame
        self.max_seq_trials = 100  # number of attempts at generating a sequence from the seed frame

        self.seq_sampler = TrainFrameSampler()
        self.transform = VideoMatteTrainAugmentation(size)

        for dataset, config in data_configs.items():
            total_frames = 0
            multiplier = config['multiplier']

            self.background_image_dir = config['bg_img_directory']
            self.background_image_files = os.listdir(config['bg_img_directory'])
            self.background_video_dir = config['bg_video_directory']
            self.background_video_clips = sorted(os.listdir(config['bg_video_directory']))
            self.background_video_frames = [sorted(os.listdir(os.path.join(config['bg_video_directory'], clip)))
                                            for clip in self.background_video_clips]
            
            self.videomatte_dir = config['fgr_video_directory']
            self.videomatte_clips = sorted(os.listdir(os.path.join(config['fgr_video_directory'], 'fgr_new')))
            self.videomatte_frames = [sorted(os.listdir(os.path.join(config['fgr_video_directory'], 'fgr_new', clip))) 
                                    for clip in self.videomatte_clips]
            self.videomatte_idx = [(clip_idx, frame_idx) 
                                for clip_idx in range(len(self.videomatte_clips)) 
                                for frame_idx in range(0, len(self.videomatte_frames[clip_idx]), seq_length)]
            
            for clip_idx in range(len(self.videomatte_clips)):
                total_frames += len(self.videomatte_frames[clip_idx])

            if local_rank == 0:
                log.info(
                    f'{dataset}: {len(self.videomatte_clips)}/{len(self.videomatte_clips)} videos will be used in {self.videomatte_dir}.'
                )
                log.info(
                    f'{dataset}: {total_frames} frames found. Multiplied to {total_frames*multiplier} frames.'
                )

        if local_rank == 0:
            log.info(f'Total number of video-frames: {total_frames*multiplier}.')

    
    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr] * self.seq_length
        return bgrs
    
    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _get_videomatte(self, idx):
        clip_idx, frame_idx = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        frame_count = len(self.videomatte_frames[clip_idx])
        fgrs, phas = [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
            with Image.open(os.path.join(self.videomatte_dir, 'fgr_new', clip, frame)) as fgr, \
                 Image.open(os.path.join(self.videomatte_dir, 'pha', clip, frame)) as pha:
                    fgr = self._downsample_if_needed(fgr.convert('RGB'))
                    pha = self._downsample_if_needed(pha.convert('L'))
            fgrs.append(fgr)
            phas.append(pha)
        return fgrs, phas
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img
    
    def _RandomResizedCrop_if_needed(self, img):
        w, h = img.size
        min_scale = min((self.size*self.size) / (w*h), 1)
        max_scale = max(min_scale, 0.5)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(self.size, self.size), scale=(min_scale, max_scale), ratio=(0.9, 1.1))
        ])

        transformed_image = transform(img)
        return transformed_image
    
    def box_blur(self, img, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        padded_img = np.pad(img, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='constant', constant_values=0)
        blurred_img = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                blurred_img[i, j] = np.sum(padded_img[i:i + kernel_size, j:j + kernel_size] * kernel)

        return blurred_img
    
    def add_noise(self, img):
        img /= 255
        grain_size = random.random() * 3 + 1  # range 1 ~ 4
        H, W = img.shape
        noise = np.random.randn(round(H / grain_size), round(W / grain_size))
        noise *= random.random() * 0.2 / grain_size
        
        if grain_size != 1:
            noise = cv2.resize(noise, (H, W))
        
        img = img + noise
        img = np.clip(img, 0, 1)
        img *= 255
        return img
    
    def gen_dilate(self, alpha, min_kernel_size, max_kernel_size): 
        kernel_size = random.randint(min_kernel_size, max_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
        return dilate.astype(np.float32)

    def gen_erosion(self, alpha, min_kernel_size, max_kernel_size): 
        kernel_size = random.randint(min_kernel_size, max_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        erode = cv2.erode(fg, kernel, iterations=1)*255
        return erode.astype(np.float32)

    def _get_sample(self, idx=None):
        # pick, augment, and return a video sequence
        # We look at the sequence given by idx first, but there is no guarantee that we will use it
        if idx is None:
            idx = np.random.randint(len(self.videomatte_idx))

        info = dict()
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
        
        fgrs, phas = self._get_videomatte(idx)
        fgrs, phas, bgrs = self.transform(fgrs, phas, bgrs)
        images = fgrs * phas + bgrs * (1-phas)
        masks = np.array(phas[:,0,:,:])
        return info, images, masks

    def __getitem__(self, idx):

        info, images, masks = self._get_sample(idx)
        labels = [1]

        assert len(labels) > 0  # should not be empty at all times
        target_objects = labels

        # if there are more than max_num_obj objects, subsample them
        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))
        info['is_img'] = False

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.seq_length, self.size, self.size), dtype=np.float32)
        first_frame_gt = np.zeros((1, self.max_num_obj, self.size, self.size), dtype=np.float32)
        for i, l in enumerate(target_objects):
            this_mask = (masks > 0)
            cls_gt[this_mask] = i + 1
            first_frame_gt[0, i] = (masks[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        # Generate first-frame segmentation mask
        pha = first_frame_gt[0,0] * 255
        msk = np.zeros_like(pha)
        msk[pha > 50] = 255

        if random.random() < 0.5:
            msk = self.gen_dilate(msk, 1, 8)
        elif random.random() < 1.0:
            msk = self.gen_erosion(msk, 1, 8)
        else:
            pass

        first_frame_gt = (msk/255).reshape(1, 1, pha.shape[0], pha.shape[1])
        first_frame_gt = first_frame_gt.astype(np.float32)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt':  np.expand_dims(masks, 1),  
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videomatte_idx)

class VideoMatteTrainAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )

class VideoMatteValidAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            prob_pause=0,
        )