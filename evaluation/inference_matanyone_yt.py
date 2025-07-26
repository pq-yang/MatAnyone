import os
import cv2
import tqdm
import imageio
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from hugging_face.tools.download_util import load_file_from_url
from matanyone.utils.inference_utils import gen_dilate, gen_erosion, read_frame_from_videos

from matanyone.inference.inference_core import InferenceCore
from matanyone.utils.get_default_model import get_matanyone_model
from matanyone.utils.device import get_default_device, safe_autocast_decorator

import warnings
warnings.filterwarnings("ignore")

device = get_default_device()

@torch.inference_mode()
@safe_autocast_decorator()
def main(input_path, mask_path, output_path, ckpt_path, n_warmup=10, r_erode=10, r_dilate=10, suffix="", save_image=False, max_size=-1):

    video_name = os.path.basename(os.path.dirname(input_path))

    # download ckpt for the first inference
    pretrain_model_url = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
    ckpt_path = load_file_from_url(pretrain_model_url, 'pretrained_models')
    
    # load MatAnyone model
    matanyone = get_matanyone_model(ckpt_path, device)

    # init inference processor
    processor = InferenceCore(matanyone, cfg=matanyone.cfg)

    # inference parameters
    r_erode = int(r_erode)
    r_dilate = int(r_dilate)
    n_warmup = int(n_warmup)
    max_size = int(max_size)

    # load input frames
    vframes, fps, length, _ = read_frame_from_videos(input_path)
    repeated_frames = vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1) # repeat the first frame for warmup
    vframes = torch.cat([repeated_frames, vframes], dim=0).float()
    length += n_warmup  # update length

    # resize if needed
    if max_size > 0:
        h, w = vframes.shape[-2:]
        min_side = min(h, w)
        if min_side > max_size:
            new_h = int(h / min_side * max_size)
            new_w = int(w / min_side * max_size)

        vframes = F.interpolate(vframes, size=(new_h, new_w), mode="area")
        
    # set output paths
    os.makedirs(output_path, exist_ok=True)
    if suffix != "":
        video_name = f'{video_name}_{suffix}'
    if save_image:
        os.makedirs(f'{output_path}/{video_name}', exist_ok=True)
        os.makedirs(f'{output_path}/{video_name}/pha', exist_ok=True)
        os.makedirs(f'{output_path}/{video_name}/fgr', exist_ok=True)

    # load the first-frame mask
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)

    bgr = (np.array([120, 255, 155], dtype=np.float32)/255).reshape((1, 1, 3)) # green screen to paste fgr
    objects = [1]

    # [optional] erode & dilate
    if r_dilate > 0:
        mask = gen_dilate(mask, r_dilate, r_dilate)
    if r_erode > 0:
        mask = gen_erosion(mask, r_erode, r_erode)

    mask = torch.from_numpy(mask).float().to(device)

    if max_size > 0:  # resize needed
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest")
        mask = mask[0,0]

    # inference start
    phas = []
    fgrs = []
    for ti in tqdm.tqdm(range(length)):
        # load the image as RGB; normalization is done within the model
        image = vframes[ti]

        image_np = np.array(image.permute(1,2,0))       # for output visualize
        image = (image / 255.).float().to(device)         # for network input

        if ti == 0:
            output_prob = processor.step(image, mask, objects=objects)      # encode given mask
            output_prob = processor.step(image, first_frame_pred=True)      # first frame for prediction
        else:
            if ti <= n_warmup:
                output_prob = processor.step(image, first_frame_pred=True)  # reinit as the first frame for prediction
            else:
                output_prob = processor.step(image)

        # convert output probabilities to alpha matte
        mask = processor.output_prob_to_mask(output_prob)

        # visualize prediction
        pha = mask.unsqueeze(2).cpu().numpy()
        com_np = image_np / 255. * pha + bgr * (1 - pha)
        
        # DONOT save the warmup frame
        if ti > (n_warmup-1):
            com_np = np.round(np.clip(com_np * 255.0, 0, 255)).astype(np.uint8)
            pha = np.round(np.clip(pha * 255.0, 0, 255)).astype(np.uint8)
            fgrs.append(com_np)
            phas.append(pha)
            if save_image:
                cv2.imwrite(f'{output_path}/{video_name}/pha/{str(ti-n_warmup).zfill(4)}.png', pha)
                cv2.imwrite(f'{output_path}/{video_name}/fgr/{str(ti-n_warmup).zfill(4)}.png', com_np[...,[2,1,0]])

    phas = np.array(phas)
    fgrs = np.array(fgrs)

    imageio.mimwrite(f'{output_path}/{video_name}_fgr.mp4', fgrs, fps=fps, quality=7)
    imageio.mimwrite(f'{output_path}/{video_name}_pha.mp4', phas, fps=fps, quality=7)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default="inputs/video/test-sample1.mp4", help='Path of the input video or frame folder.')
    parser.add_argument('-m', '--mask_path', type=str, default="inputs/mask/test-sample1.png", help='Path of the first-frame segmentation mask.')
    parser.add_argument('-o', '--output_path', type=str, default="results/", help='Output folder. Default: results')
    parser.add_argument('-c', '--ckpt_path', type=str, default="pretrained_models/matanyone.pth", help='Path of the MatAnyone model.')
    parser.add_argument('-w', '--warmup', type=str, default="10", help='Number of warmup iterations for the first frame alpha prediction.')
    parser.add_argument('-e', '--erode_kernel', type=str, default="10", help='Erosion kernel on the input mask.')
    parser.add_argument('-d', '--dilate_kernel', type=str, default="10", help='Dilation kernel on the input mask.')
    parser.add_argument('--suffix', type=str, default="", help='Suffix to specify different target when saving, e.g., target1.')
    parser.add_argument('--save_image', action='store_true', default=False, help='Save output frames. Default: False')
    parser.add_argument('--max_size', type=str, default="-1", help='When positive, the video will be downsampled if min(w, h) exceeds. Default: -1 (means no limit)')

    
    args = parser.parse_args()

    main(input_path=args.input_path, \
         mask_path=args.mask_path, \
         output_path=args.output_path, \
         ckpt_path=args.ckpt_path, \
         n_warmup=args.warmup, \
         r_erode=args.erode_kernel, \
         r_dilate=args.dilate_kernel, \
         suffix=args.suffix, \
         save_image=args.save_image, \
         max_size=args.max_size)
