from typing import List, Dict
from omegaconf import DictConfig
from collections import defaultdict
import torch
import torch.nn.functional as F

from matanyone.utils.point_features import calculate_uncertainty, point_sample, get_uncertain_point_coords_with_randomness
from matanyone.utils.tensor_utils import cls_to_one_hot


@torch.jit.script
def ce_loss(logits: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    # logits: T*C*num_points
    loss = F.cross_entropy(logits, soft_gt, reduction='none')
    # sum over temporal dimension
    return loss.sum(0).mean()


@torch.jit.script
def dice_loss(mask: torch.Tensor, soft_gt: torch.Tensor) -> torch.Tensor:
    # mask: T*C*num_points
    # soft_gt: T*C*num_points
    # ignores the background
    mask = mask[:, 1:].flatten(start_dim=2)
    gt = soft_gt[:, 1:].float().flatten(start_dim=2)
    numerator = 2 * (mask * gt).sum(-1)
    denominator = mask.sum(-1) + gt.sum(-1)
    
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum(0).mean()

def seg2trimap(gt_mask, kernel_size=None):
    """
    Convert segmentation mask to trimap with adaptive kernel size.
    
    Args:
        gt_mask: Input mask tensor (B, C, H, W) with values in [0, 1] or [0, 255]
        kernel_size: Optional fixed kernel size. If None, will be calculated adaptively.
    
    Returns:
        trimap: Trimap with values 0 (background), 0.5 (unknown), 1 (foreground)
    """
    # Calculate adaptive kernel size based on non-zero area
    if kernel_size is None:
        # Count non-zero pixels
        non_zero_mask = (gt_mask > 0).float()
        non_zero_area = non_zero_mask.sum()
        
        # Calculate adaptive kernel size based on area
        # Use square root of area as base, with minimum and maximum bounds
        if non_zero_area > 0:
            # Scale factor to ensure uncertain region covers all intermediate values
            # Larger area needs larger kernel to create sufficient uncertain region
            adaptive_size = max(9, min(21, int(torch.sqrt(non_zero_area).item() * 0.1)))
        else:
            adaptive_size = 9  # Default fallback
        
        kernel_size = adaptive_size
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create the structuring element for morphological operations
    padding = kernel_size // 2
    struct_elem = torch.ones(1, 1, kernel_size, kernel_size, device=gt_mask.device)

    # Dilation to expand the foreground
    dilated_mask = F.conv2d(gt_mask, struct_elem, padding=padding)
    dilated_mask = (dilated_mask > 0).float()
    
    # Erosion to shrink the foreground
    eroded_mask = F.conv2d(gt_mask, struct_elem, padding=padding)
    eroded_mask = (eroded_mask == kernel_size * kernel_size).float()
    
    # Trimap generation: 0 = background, 0.5 = unknown, 1 = foreground
    trimap = torch.zeros_like(gt_mask)
    trimap[eroded_mask == 1] = 1  # Foreground
    trimap[(dilated_mask - eroded_mask) > 0] = 0.5  # Unknown

    return trimap

def scaled_ddc_loss(image, alpha, kernel_size):
    b, c, h, w = image.shape
    unfold_image = F.unfold(image, kernel_size= kernel_size, padding=kernel_size // 2).view(b, c, kernel_size ** 2, h, w)

    ### approximate |F-B| by taking average
    # Calculate top k smallest and largest values from the patches
    # 1. convert from RGB to YCrCb
    R = unfold_image[:, 0, :, :, :]  # (b, ksq, h, w)
    G = unfold_image[:, 1, :, :, :]  # (b, ksq, h, w)
    B = unfold_image[:, 2, :, :, :]  # (b, ksq, h, w)
    # Calculate Y (Luminance)
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # (b, ksq, h, w)
    # 2. select top k values based on Y
    top_k_min_vals, _ = torch.topk(Y, k=kernel_size, dim=1, largest=False) # b k h w
    top_k_max_vals, _ = torch.topk(Y, k=kernel_size, dim=1, largest=True)  # b k h w
    
    # Calculate the mean of top k smallest and largest values
    top_k_min_mean = top_k_min_vals.mean(dim=1)  # Shape: (b, h, w)
    top_k_max_mean = top_k_max_vals.mean(dim=1)  # Shape: (b, h, w)
    
    # Difference between the mean of top k max and min values
    image_diff = top_k_max_mean - top_k_min_mean

    image_dist = torch.norm(image.view(b, c, 1, h , w) - unfold_image, 2, dim=1)
    image_dist, indices = torch.topk(image_dist, k=kernel_size, dim=1, largest=False)
    unfold_alpha = F.unfold(alpha, kernel_size= kernel_size, padding=kernel_size // 2).view(b, kernel_size ** 2, h, w)
    alpha_dist = torch.gather(alpha-unfold_alpha, dim=1, index=indices)
    scaled_alpha_dist = alpha_dist * image_diff.view(b, 1, h, w)
    return F.l1_loss(image_dist, scaled_alpha_dist)

# Loss computation for core supervision pass
class CoreSupervisionLossComputer:
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__()
        self.point_supervision = stage_cfg.point_supervision
        self.num_points = stage_cfg.train_num_points
        self.oversample_ratio = stage_cfg.oversample_ratio
        self.importance_sample_ratio = stage_cfg.importance_sample_ratio

        self.sensory_weight = cfg.model.aux_loss.sensory.weight
        self.query_weight = cfg.model.aux_loss.query.weight

    def compute(self, data: Dict[str, torch.Tensor],
                num_objects: List[int]) -> Dict[str, torch.Tensor]:
        batch_size, num_frames = data['rgb'].shape[:2]
        losses = defaultdict(float)
        t_range = range(1, num_frames)

        for bi in range(batch_size):

            imgs = data['rgb'][bi] # T C H W, [0, 1]
            logits = torch.stack([data[f'logits_{ti}'][bi, 1 : (num_objects[bi] + 1)] for ti in t_range], dim=0)  # remove background
            cls_gt = data['cls_gt'][bi, 1:].float()  # remove gt for the first frame

            trimap = seg2trimap(cls_gt)
            known_region_mask = ((trimap == 0) | (trimap == 1)).float()

            first_logits = data[f'logits_0'][bi, 1 : (num_objects[bi] + 1)].unsqueeze(0)  # remove background
            first_cls_gt = data['cls_gt'][bi, 0:1].float()  # remove gt for the first frame

            first_trimap = seg2trimap(first_cls_gt)
            first_known_region_mask = ((first_trimap == 0) | (first_trimap == 1)).float()

            # temporal uncertainty ground-truth
            spar_gt = torch.abs(data['cls_gt'][bi, 1:] - data['cls_gt'][bi, :-1]).float()
            spar_gt = (F.interpolate(spar_gt, scale_factor=1/16, mode='area') > 0) + 0
            
            losses['scaled_DDC_loss'] = scaled_ddc_loss(imgs, torch.cat((first_logits, logits), dim=0), 11) / batch_size * 10 * 1.15  # empirically hardcode
            losses['known_l1'] += F.l1_loss(logits*known_region_mask, cls_gt*known_region_mask, reduction="sum") / torch.sum(known_region_mask) / batch_size
            losses['first_known_l1'] += F.l1_loss(first_logits*first_known_region_mask, first_cls_gt*first_known_region_mask, reduction="sum") / torch.sum(first_known_region_mask) / batch_size * 5

            aux = [data[f'aux_{ti}'] for ti in range(0, num_frames)]
            cls_gt = data['cls_gt'][bi, :].float()  # DONOT remove gt for the first frame
            full_known_region_mask = torch.cat((first_known_region_mask, known_region_mask), dim=0)

            if 'sensory_logits' in aux[0]:
                sensory_log = torch.stack(
                    [a['sensory_logits'][bi, 1 : num_objects[bi] + 1] for a in aux], dim=0)                       # remove background
                sensory_log = F.interpolate(sensory_log, size=cls_gt.shape[-2:], mode='bilinear')

                losses['aux_sensory_known_l1'] += F.l1_loss(sensory_log*full_known_region_mask, cls_gt*full_known_region_mask, reduction="sum") / torch.sum(full_known_region_mask) / batch_size * self.sensory_weight
            
            if 'q_logits' in aux[0]:
                num_levels = aux[0]['q_logits'].shape[2]

                for l in range(num_levels):
                    query_log = torch.stack(
                        [a['q_logits'][bi, 1 : num_objects[bi] + 1, l] for a in aux], dim=0)                      # remove background
                    query_log = F.interpolate(query_log, size=cls_gt.shape[-2:], mode='bilinear')

                    losses[f'aux_query_l{l}_known_l1'] += F.l1_loss(query_log*full_known_region_mask, cls_gt*full_known_region_mask, reduction="sum") / torch.sum(full_known_region_mask) / batch_size * self.query_weight
            
            # compute loss for uncertainty prediction for video data only
            if (not data['info']['is_img'][0]) and data[f'ts_1'] is not None:
                ts = [data[f'ts_{ti}'] for ti in t_range]
                ts_log = torch.stack(
                    [s['logits'][bi, :num_objects[bi] + 1] for s in ts], dim=0)
                ts_log = F.interpolate(ts_log, size=spar_gt.shape[-2:], mode='bilinear')
                loss_bce = F.binary_cross_entropy_with_logits(ts_log, spar_gt.float(), reduction='mean')
                losses['ts_bce'] += loss_bce / batch_size

        losses['total_loss'] = sum(losses.values())

        return losses

# Loss computation for segmentation pass
class LossComputer:
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__()
        self.point_supervision = stage_cfg.point_supervision
        self.num_points = stage_cfg.train_num_points
        self.oversample_ratio = stage_cfg.oversample_ratio
        self.importance_sample_ratio = stage_cfg.importance_sample_ratio

        self.sensory_weight = cfg.model.aux_loss.sensory.weight
        self.query_weight = cfg.model.aux_loss.query.weight

    def mask_loss(self, logits: torch.Tensor,
                  soft_gt: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        assert self.point_supervision

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                logits, lambda x: calculate_uncertainty(x), self.num_points, self.oversample_ratio,
                self.importance_sample_ratio)
            # get gt labels
            point_labels = point_sample(soft_gt, point_coords, align_corners=False)
        point_logits = point_sample(logits, point_coords, align_corners=False)
        # point_labels and point_logits: B*C*num_points

        loss_ce = ce_loss(point_logits, point_labels)
        loss_dice = dice_loss(point_logits.softmax(dim=1), point_labels)

        return loss_ce, loss_dice

    def compute(self, data: Dict[str, torch.Tensor],
                num_objects: List[int]) -> Dict[str, torch.Tensor]:
        batch_size, num_frames = data['rgb'].shape[:2]
        losses = defaultdict(float)
        t_range = range(1, num_frames)
            
        for bi in range(batch_size):
            logits = torch.stack([data[f'logits_{ti}'][bi, :num_objects[bi] + 1] for ti in t_range],
                                 dim=0)
            cls_gt = data['cls_gt'][bi, 1:]  # remove gt for the first frame
            soft_gt = cls_to_one_hot(cls_gt, num_objects[bi])

            first_logits = data[f'logits_0'][bi, :num_objects[bi] + 1].unsqueeze(0)
            first_cls_gt = data['cls_gt'][bi, 0:1]  # gt for the first frame
            first_soft_gt = cls_to_one_hot(first_cls_gt, num_objects[bi])

            # temporal uncertainty ground-truth
            spar_gt = torch.abs(data['cls_gt'][bi, 1:] - data['cls_gt'][bi, :-1]) # hardcode
            spar_gt = (F.interpolate(spar_gt.float(), scale_factor=1/16, mode='area') > 0) + 0

            loss_ce, loss_dice = self.mask_loss(logits, soft_gt)
            losses['loss_ce'] += loss_ce / batch_size
            losses['loss_dice'] += loss_dice / batch_size

            first_loss_ce, first_loss_dice = self.mask_loss(first_logits, first_soft_gt)
            losses['first_loss_ce'] += first_loss_ce / batch_size * 5
            losses['first_loss_dice'] += first_loss_dice / batch_size * 5

            
            aux = [data[f'aux_{ti}'] for ti in range(0, num_frames)]
            cls_gt = data['cls_gt'][bi, :]  # DONOT remove gt for the first frame
            soft_gt = cls_to_one_hot(cls_gt, num_objects[bi])

            if 'sensory_logits' in aux[0]:
                sensory_log = torch.stack(
                    [a['sensory_logits'][bi, :num_objects[bi] + 1] for a in aux], dim=0)
                loss_ce, loss_dice = self.mask_loss(sensory_log, soft_gt)
                losses['aux_sensory_ce'] += loss_ce / batch_size * self.sensory_weight
                losses['aux_sensory_dice'] += loss_dice / batch_size * self.sensory_weight
            if 'q_logits' in aux[0]:
                num_levels = aux[0]['q_logits'].shape[2]

                for l in range(num_levels):
                    query_log = torch.stack(
                        [a['q_logits'][bi, :num_objects[bi] + 1, l] for a in aux], dim=0)
                    loss_ce, loss_dice = self.mask_loss(query_log, soft_gt)
                    losses[f'aux_query_ce_l{l}'] += loss_ce / batch_size * self.query_weight
                    losses[f'aux_query_dice_l{l}'] += loss_dice / batch_size * self.query_weight
            
            # compute loss for uncertainty prediction for video data only
            if (not data['info']['is_img'][0]) and data[f'ts_1'] is not None:
                ts = [data[f'ts_{ti}'] for ti in t_range]
                ts_log = torch.stack(
                    [s['logits'][bi, :num_objects[bi] + 1] for s in ts], dim=0)
                ts_log = F.interpolate(ts_log, size=spar_gt.shape[-2:], mode='bilinear')
                loss_bce = F.binary_cross_entropy_with_logits(ts_log, spar_gt.float(), reduction='mean')
                losses['ts_bce'] += loss_bce / batch_size

        losses['total_loss'] = sum(losses.values())

        return losses

# Loss computation for matting pass
class MatLossComputer:
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__()
        self.point_supervision = stage_cfg.point_supervision
        self.num_points = stage_cfg.train_num_points
        self.oversample_ratio = stage_cfg.oversample_ratio
        self.importance_sample_ratio = stage_cfg.importance_sample_ratio

        self.sensory_weight = cfg.model.aux_loss.sensory.weight
        self.query_weight = cfg.model.aux_loss.query.weight

    def mask_loss(self, logits: torch.Tensor,
                  soft_gt: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        assert self.point_supervision

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                logits, lambda x: calculate_uncertainty(x), self.num_points, self.oversample_ratio,
                self.importance_sample_ratio)
            # get gt labels
            point_labels = point_sample(soft_gt, point_coords, align_corners=False)
        point_logits = point_sample(logits, point_coords, align_corners=False)
        # point_labels and point_logits: B*C*num_points

        loss_ce = ce_loss(point_logits, point_labels)
        loss_dice = dice_loss(point_logits.softmax(dim=1), point_labels)

        return loss_ce, loss_dice

    def compute(self, data: Dict[str, torch.Tensor],
                num_objects: List[int]) -> Dict[str, torch.Tensor]:
        batch_size, num_frames = data['rgb'].shape[:2]
        losses = defaultdict(float)
        t_range = range(1, num_frames)

        for bi in range(batch_size):

            logits = torch.stack([data[f'logits_{ti}'][bi, 1 : (num_objects[bi] + 1)] for ti in t_range], dim=0)  # remove background
            cls_gt = data['cls_gt'][bi, 1:]  # remove gt for the first frame

            first_logits = data[f'logits_0'][bi, 1 : (num_objects[bi] + 1)].unsqueeze(0)  # remove background
            first_cls_gt = data['cls_gt'][bi, 0:1]  # remove gt for the first frame

            # temporal uncertainty ground-truth
            spar_gt = torch.abs(data['cls_gt'][bi, 1:] - data['cls_gt'][bi, :-1]) # hardcode
            if not data['info']['is_img'][0]:
                spar_gt = (F.interpolate(spar_gt, scale_factor=1/16, mode='area') > 0.001) + 0   # video
            else:
                spar_gt = (F.interpolate(spar_gt, scale_factor=1/16, mode='area') > 0) + 0       # img
            
            losses['pha_l1'] += F.l1_loss(logits, cls_gt) / batch_size 
            losses['pha_laplacian'] += laplacian_loss(logits, cls_gt) / batch_size 
            losses['pha_coherence'] += F.mse_loss(logits[1:] - logits[:-1],
                    cls_gt[1:] - cls_gt[:-1]) / batch_size         

            losses['first_pha_l1'] += F.l1_loss(first_logits, first_cls_gt) / batch_size * 5
            losses['first_pha_laplacian'] += laplacian_loss(first_logits, first_cls_gt) / batch_size * 5

            aux = [data[f'aux_{ti}'] for ti in range(0, num_frames)]
            cls_gt = data['cls_gt'][bi, :]  # DONOT remove gt for the first frame

            if 'sensory_logits' in aux[0]:
                sensory_log = torch.stack(
                    [a['sensory_logits'][bi, 1 : num_objects[bi] + 1] for a in aux], dim=0)                       # remove background
                sensory_log = F.interpolate(sensory_log, size=cls_gt.shape[-2:], mode='bilinear')

                losses['aux_sensory_pha_l1'] += F.l1_loss(sensory_log, cls_gt) / batch_size * self.sensory_weight
                losses['aux_sensory_pha_laplacian'] += laplacian_loss(sensory_log, cls_gt) / batch_size * self.sensory_weight
            if 'q_logits' in aux[0]:
                num_levels = aux[0]['q_logits'].shape[2]

                for l in range(num_levels):
                    query_log = torch.stack(
                        [a['q_logits'][bi, 1 : num_objects[bi] + 1, l] for a in aux], dim=0)                      # remove background
                    query_log = F.interpolate(query_log, size=cls_gt.shape[-2:], mode='bilinear')

                    losses[f'aux_query_l{l}_pha_l1'] += F.l1_loss(query_log, cls_gt) / batch_size * self.query_weight
                    losses[f'aux_query_l{l}_pha_laplacian'] += laplacian_loss(query_log, cls_gt) / batch_size * self.query_weight
            
            # compute loss for uncertainty prediction for video data only
            if (not data['info']['is_img'][0]) and data[f'ts_1'] is not None:
                ts = [data[f'ts_{ti}'] for ti in t_range]
                ts_log = torch.stack(
                    [s['logits'][bi, :num_objects[bi] + 1] for s in ts], dim=0)
                ts_log = F.interpolate(ts_log, size=spar_gt.shape[-2:], mode='bilinear')
                loss_bce = F.binary_cross_entropy_with_logits(ts_log, spar_gt.float(), reduction='mean')
                losses['ts_bce'] += loss_bce / batch_size

        losses['total_loss'] = sum(losses.values())

        return losses

# ----------------------------------------------------------------------------- Laplacian Loss

def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]


