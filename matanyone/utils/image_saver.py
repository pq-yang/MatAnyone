import cv2
import numpy as np

import torch
from collections import defaultdict
import torch.nn.functional as F


def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np


def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np


def detach_to_cpu(x):
    return x.detach().cpu()


def transpose_np(x):
    return np.transpose(x, [1, 2, 0])


def tensor_to_gray_im(x):
    x = detach_to_cpu(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x


def tensor_to_im(x):
    x = detach_to_cpu(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x


# Predefined key <-> caption dict
key_captions = {
    'im': 'Image',
    'gt': 'GT',
}
"""
Return an image array with captions
keys in dictionary will be used as caption if not provided
values should contain lists of cv2 images
"""


def get_image_array(images, grid_shape, captions={}, seg_pass=False):
    h, w = grid_shape
    cate_counts = len(images)
    rows_counts = len(next(iter(images.values())))

    font = cv2.FONT_HERSHEY_SIMPLEX

    output_image = np.zeros([w * cate_counts, h * (rows_counts + 1), 3], dtype=np.uint8)
    col_cnt = 0
    for k, v in images.items():

        # Default as key value itself
        caption = captions.get(k, k)

        # Handles new line character
        dy = 40
        for i, line in enumerate(caption.split('\n')):
            cv2.putText(output_image, line, (10, col_cnt * w + 100 + i * dy), font, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # Put images
        for row_cnt, img in enumerate(v):
            im_shape = img.shape
            if len(im_shape) == 2:
                img = img[..., np.newaxis]

            img = (img * 255).astype('uint8')

            output_image[(col_cnt + 0) * w:(col_cnt + 1) * w,
                         (row_cnt + 1) * h:(row_cnt + 2) * h, :] = img

        col_cnt += 1

    return output_image


def base_transform(im, size):
    im = tensor_to_np_float(im)
    if len(im.shape) == 3:
        im = im.transpose((1, 2, 0))
    else:
        im = im[:, :, None]

    # Resize
    if im.shape[1] != size:
        im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)

    return im.clip(0, 1)


def im_transform(im, size):
    return base_transform(detach_to_cpu(im), size=size)


def mask_transform(mask, size):
    return base_transform(detach_to_cpu(mask), size=size)


def logits_transform(mask, size):
    return base_transform(detach_to_cpu(torch.sigmoid(mask)), size=size)


def add_attention(mask, pos):
    mask = mask[:, :, None].repeat(3, axis=2)
    pos = (pos + 1) / 2
    for i in range(pos.shape[0]):
        y = int(pos[i][0] * mask.shape[0])
        x = int(pos[i][1] * mask.shape[1])
        y = max(min(y, mask.shape[0] - 1), 0)
        x = max(min(x, mask.shape[1] - 1), 0)
        # mask[y, x, :] = (255, 0, 0)
        cv2.circle(mask, (x, y), 5, (1, 0, 0), -1)
    return mask

def vis(images, size, num_objects, seg_pass=False, num_sample=2):
    req_images = defaultdict(list)

    b, t = images['rgb'].shape[:2]

    # limit the number of images saved
    b = min(num_sample, b)

    # find max num objects
    max_num_objects = max(num_objects[:b])

    for bi in range(b):
        if images[f'ts_1'] is not None:
            spar_gt = torch.abs(images['cls_gt'][bi, 1:] - images['cls_gt'][bi, :-1]) # hardcode
            spar_gt = (F.interpolate(spar_gt.float(), scale_factor=1/16, mode='bilinear') > 0) + 0

        for ti in range(t):
            req_images['RGB'].append(im_transform(images['rgb'][bi, ti], size))

            for oi in range(max_num_objects):
                if ti == 0 or oi >= num_objects[bi]:
                    logits = mask_transform(images[f'logits_{ti}'][bi][1], size)
                    req_images[f'Mask Pred'].append(logits)
                    if images[f'ts_1'] is not None:
                        req_images[f'Uncert Pred'].append(
                            mask_transform(images['first_frame_gt'][bi][0, oi], size))
                else:
                    logits = mask_transform(images[f'logits_{ti}'][bi][1], size)
                    req_images[f'Mask Pred'].append(logits)
                    if images[f'ts_{ti}'] is not None:
                        logits = mask_transform(images[f'ts_{ti}']['logits'][bi][0], size)
                        req_images[f'Uncert Pred'].append(logits)
                if seg_pass:
                    req_images[f'Mask GT'].append(
                        mask_transform(images['cls_gt'][bi, ti, 0] == (oi + 1), size))
                    if images[f'ts_1'] is not None:
                        if ti == 0:
                            req_images[f'Uncert GT'].append(
                                mask_transform(images['cls_gt'][bi, ti, 0] == (oi + 1), size))
                        else:
                            req_images[f'Uncert GT'].append(
                                mask_transform(spar_gt[ti-1, 0], size))
                else:
                    req_images[f'Mask GT'].append(
                        mask_transform(images['cls_gt'][bi, ti, 0], size))
                    if images[f'ts_1'] is not None:
                        if ti == 0:
                            req_images[f'Uncert GT'].append(
                                mask_transform(images['cls_gt'][bi, ti, 0], size))
                        else:
                            req_images[f'Uncert GT'].append(
                                mask_transform(spar_gt[ti-1, 0], size))

    return get_image_array(req_images, size, key_captions, seg_pass)