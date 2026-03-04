import os
from os import path
import logging
import json
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import random
import cv2

from matanyone.dataset.utils import im_mean, reseed
from matanyone.dataset.tps import random_tps_warp

log = logging.getLogger()
local_rank = int(os.environ['LOCAL_RANK'])


class SyntheticVideoDataset(Dataset):
    """
    Note: data normalization happens within the model instead of here
    Generate pseudo VOS data by applying random transforms on static images.

    parameters is a list of tuples 
        (data_root, how data is structured (method 0 or 1), and an oversample multiplier)

    """
    def __init__(self, parameters, *, size=384, seq_length=3, max_num_obj=1):
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size

        self.im_list = []
        self.gt_list = []
        self.masks = {}
        for parameter in parameters:
            imgdir, anndir, annfile, multiplier = parameter
            
            if annfile is not None:
                with open(annfile) as f:
                    ann = json.load(f)['annotations']
                    ann = list(filter(lambda data: any(info['category_id'] == 1 for info in data['segments_info']), ann))
                for info in ann:
                    im_path = path.join(imgdir, info['file_name'][:-3] + 'jpg')
                    self.masks[im_path] = info['segments_info']
                    self.im_list.append(im_path)
                    self.gt_list.append(path.join(anndir, info['file_name']))
            else:
                self.im_list.extend(
                    [path.join(imgdir, im) for im in sorted(os.listdir(imgdir))] * multiplier)
                self.gt_list.extend(
                    [path.join(anndir, im) for im in sorted(os.listdir(anndir))] * multiplier)

        if local_rank == 0:
            log.info(f'SyntheticVideoDataset: {len(self.im_list)} images found in total.')

        # The frame transforms are the same for each of the pairs,
        # but different for different pairs in the sequence
        self.frame_image_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0),
        ])

        self.frame_image_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20,
                                    scale=(0.5, 2.0),
                                    shear=10,
                                    interpolation=InterpolationMode.BILINEAR,
                                    fill=im_mean),
            transforms.Resize(self.size, InterpolationMode.BILINEAR),
            transforms.RandomCrop((self.size, self.size), pad_if_needed=True, fill=im_mean),
        ])

        self.frame_mask_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20,
                                    scale=(0.5, 2.0),
                                    shear=10,
                                    interpolation=InterpolationMode.NEAREST,
                                    fill=0),
            transforms.Resize(self.size, InterpolationMode.NEAREST),
            transforms.RandomCrop((self.size, self.size), pad_if_needed=True, fill=0),
        ])

        # The sequence transforms are the same for all pairs in the sampled sequence
        self.sequence_image_only_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
            transforms.RandomGrayscale(0.05),
        ])

        self.sequence_image_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.5, 2.0), fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.sequence_mask_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.5, 2.0), fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        self.output_image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.output_mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _get_sample(self, idx):
        im = Image.open(self.im_list[idx]).convert('RGB')
        if self.im_list[idx] in self.masks:
            with Image.open(self.gt_list[idx]) as ann:
                ann.load()
            ann = np.array(ann, copy=False).astype(np.int32)
            ann = ann[:, :, 0] + 256 * ann[:, :, 1] + 256 * 256 * ann[:, :, 2]
            seg = np.zeros(ann.shape, np.uint8)
            
            for segments_info in self.masks[self.im_list[idx]]:
                if segments_info['category_id'] in [1, 27, 32]: # person, backpack, tie
                    seg[ann == segments_info['id']] = 255
            gt = Image.fromarray(seg).convert('L')
        else:
            gt = Image.open(self.gt_list[idx]).convert('L')

        sequence_seed = np.random.randint(2147483647)

        images = []
        masks = []
        for _ in range(self.seq_length):
            reseed(sequence_seed)
            this_im = self.sequence_image_dual_transform(im)
            this_im = self.sequence_image_only_transform(this_im)
            reseed(sequence_seed)
            this_gt = self.sequence_mask_dual_transform(gt)

            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            this_im = self.frame_image_dual_transform(this_im)
            this_im = self.frame_image_transform(this_im)
            reseed(pairwise_seed)
            this_gt = self.frame_mask_dual_transform(this_gt)

            # Use TPS only some of the times
            # Not because TPS is bad -- just that it is too slow and I need to speed up data loading
            if np.random.rand() < 0.33:
                this_im, this_gt = random_tps_warp(this_im, this_gt, scale=0.02)

            this_im = self.output_image_transform(this_im)
            this_gt = self.output_mask_transform(this_gt)

            images.append(this_im)
            masks.append(this_gt)

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)

        return images, masks.numpy()
    
    def gen_dilate(self, alpha, min_kernel_size, max_kernel_size): 
        kernel_size = random.randint(min_kernel_size, max_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
        dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
        return dilate.astype(np.int64)

    def gen_erosion(self, alpha, min_kernel_size, max_kernel_size): 
        kernel_size = random.randint(min_kernel_size, max_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        erode = cv2.erode(fg, kernel, iterations=1)*255
        return erode.astype(np.int64)

    def __getitem__(self, idx):
        additional_objects = np.random.randint(self.max_num_obj)
        indices = [idx, *np.random.randint(self.__len__(), size=additional_objects)]

        # Sample from multiple images and merge them together onto a training sample
        merged_images = None
        merged_masks = np.zeros((self.seq_length, self.size, self.size), dtype=np.int64)

        for i, list_id in enumerate(indices):
            images, masks = self._get_sample(list_id)
            if merged_images is None:
                merged_images = images
            else:
                merged_images = merged_images * (1 - masks) + images * masks
            merged_masks[masks[:, 0] > 0.5] = (i + 1)

        masks = merged_masks

        labels = np.unique(masks[0])
        # Remove background
        labels = labels[labels != 0]
        target_objects = labels.tolist()

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.seq_length, self.size, self.size), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, self.size, self.size), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks == l)
            cls_gt[this_mask] = i + 1
            first_frame_gt[0, i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        info = {}
        info['name'] = self.im_list[idx]
        info['num_objects'] = max(1, len(target_objects))
        info['is_img'] = True

        # Generate first-frame segmentation mask
        msk = first_frame_gt[0,0] * 255
        
        if random.random() < 0.5:
            msk = self.gen_dilate(msk, 1, 8)
        elif random.random() < 1.0:
            msk = self.gen_erosion(msk, 1, 8)
        else:
            pass

        first_frame_gt = (msk/255).reshape(1, 1, msk.shape[0], msk.shape[1])
        first_frame_gt = first_frame_gt.astype(np.int64)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': merged_images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info
        }

        return data

    def __len__(self):
        return len(self.im_list)