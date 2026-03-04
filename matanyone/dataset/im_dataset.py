"""Modified from https://github.com/PeterL1n/RobustVideoMatting/blob/master/dataset/imagematte.py
"""
import os
from os import path
import logging
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.transforms import functional

from matanyone.dataset.augmentation import MotionAugmentation, TrainFrameSampler

log = logging.getLogger()
local_rank = int(os.environ['LOCAL_RANK'])

class ImageMatteDataset(Dataset):
    def __init__(self,
                 data_configs, seq_length=3, max_num_obj=3, size=480, merge_probability=0.0):
        
        self.configs = data_configs
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size
        self.merge_probability = merge_probability

        self.max_crop_trials = 5  # number of attempts at cropping a single frame
        self.max_seed_trials = 5  # number of attempts at changing the initial seed frame
        self.max_seq_trials = 100 # number of attempts at generating a sequence from the seed frame

        self.seq_sampler = TrainFrameSampler()
        self.transform = ImageMatteAugmentation(size)

        for dataset, config in data_configs.items():
            total_frames = 0

            multiplier = config['multiplier']
            self.imagematte_dir = config["fgr_img_directory"]
            self.imagematte_files = os.listdir(os.path.join(self.imagematte_dir, 'fgr'))
            self.background_image_dir = config["bg_img_directory"]
            self.background_image_files = os.listdir(self.background_image_dir)
            self.background_video_dir = config["bg_video_directory"]
            self.background_video_clips = os.listdir(self.background_video_dir)
            self.background_video_frames = [sorted(os.listdir(os.path.join(self.background_video_dir, clip)))
                                            for clip in self.background_video_clips]
            
            total_frames += len(self.imagematte_files)

            if local_rank == 0:
                log.info(
                    f'{dataset}: {len(self.imagematte_files)}/{len(self.imagematte_files)} images will be used in {self.imagematte_dir}.'
                )
                log.info(
                    f'{dataset}: {total_frames} frames found. Multiplied to {total_frames*multiplier} frames.'
                )

        if local_rank == 0:
            log.info(f'Total number of image-frames: {total_frames*multiplier}.')
        
    def __len__(self):
        return max(len(self.imagematte_files), len(self.background_image_files) + len(self.background_video_clips))
    
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
            idx = np.random.randint(max(len(self.imagematte_files), len(self.background_image_files) + len(self.background_video_clips)))

        info = dict()
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
        
        fgrs, phas = self._get_imagematte(idx)
        fgrs, phas, bgrs = self.transform(fgrs, phas, bgrs)
        images = fgrs * phas + bgrs * (1-phas)
        masks = np.array(phas[:,0,:,:])
        return info, images, masks, fgrs
    
    def __getitem__(self, idx):
        info, images, masks, fgrs = self._get_sample(idx)

        labels = [1]

        assert len(labels) > 0  # should not be empty at all times
        target_objects = labels

        # if there are more than max_num_obj objects, subsample them
        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))
        info['is_img'] = True

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
        msk[pha > 200] = 255

        msk = self.gen_dilate(msk, 20, 20)     # fill the holes
        msk = self.gen_erosion(msk, 20, 25)    # mimic sam output

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
            'fgrs': fgrs
        }

        return data
    
    def _get_imagematte(self, idx):
        try:
            with Image.open(os.path.join(self.imagematte_dir, 'fgr', self.imagematte_files[idx % len(self.imagematte_files)])) as fgr, \
                Image.open(os.path.join(self.imagematte_dir, 'pha', self.imagematte_files[idx % len(self.imagematte_files)])) as pha:
                fgr = self._downsample_if_needed(fgr.convert('RGB'))
                pha = self._downsample_if_needed(pha.convert('L'))
        except:
            with Image.open(os.path.join(self.imagematte_dir, 'fgr', self.imagematte_files[idx % len(self.imagematte_files)])) as fgr, \
                Image.open(os.path.join(self.imagematte_dir, 'pha', self.imagematte_files[idx % len(self.imagematte_files)][:-3]+"png")) as pha:
                fgr = self._downsample_if_needed(fgr.convert('RGB'))
                pha = self._downsample_if_needed(pha.convert('L'))

        fgrs = [fgr] * self.seq_length
        phas = [pha] * self.seq_length
        return fgrs, phas
    
    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, self.background_image_files[random.choice(range(len(self.background_image_files)))])) as bgr:
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
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img
    

class ImageMatteAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.95,
            prob_bgr_affine=0.3,
            prob_noise=0.05,
            prob_color_jitter=0.3,
            prob_grayscale=0.03,
            prob_sharpness=0.05,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )