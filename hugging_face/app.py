import sys
sys.path.append("../")
sys.path.append("../../")

import os
import json
import time
import psutil
import ffmpeg
import imageio
import argparse
from PIL import Image

import cv2
import torch
import numpy as np
import gradio as gr
 
from tools.painter import mask_painter
from tools.interact_tools import SamControler
from tools.misc import get_device
from tools.download_util import load_file_from_url

from matanyone_wrapper import matanyone
from matanyone.utils.get_default_model import get_matanyone_model
from matanyone.inference.inference_core import InferenceCore

import warnings
warnings.filterwarnings("ignore")

# * Memory management functions for GPU optimization
def move_sam_to_cpu():
    """Move SAM model to CPU to free GPU memory for matting"""
    global model
    print("Moving SAM to CPU...")
    try:
        model.samcontroler.sam_controler.model.cpu()
        torch.cuda.empty_cache()
        print("SAM moved to CPU successfully")
    except Exception as e:
        print(f"Error moving SAM to CPU: {e}")

def move_sam_to_gpu():
    """Move SAM model back to GPU after matting"""
    global model
    print("Moving SAM to GPU...")
    try:
        model.samcontroler.sam_controler.model.to(model.samcontroler.sam_controler.device)
        torch.cuda.empty_cache()
        print("SAM moved to GPU successfully")
    except Exception as e:
        print(f"Error moving SAM to GPU: {e}")

def move_matting_to_cpu():
    """Move MatAnyone model to CPU to free GPU memory for SAM"""
    global matanyone_model
    print("Moving MatAnyone to CPU...")
    try:
        matanyone_model.cpu()
        torch.cuda.empty_cache()
        print("MatAnyone moved to CPU successfully")
    except Exception as e:
        print(f"Error moving MatAnyone to CPU: {e}")

def move_matting_to_gpu():
    """Move MatAnyone model back to GPU for inference"""
    global matanyone_model
    print("Moving MatAnyone to GPU...")
    try:
        matanyone_model.to(args.device)
        print("MatAnyone moved to GPU successfully")
    except Exception as e:
        print(f"Error moving MatAnyone to GPU: {e}")

def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=8000, help="only useful when running gradio applications")  
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()
    
    if not args.device:
        args.device = str(get_device())

    return args 

# SAM generator
class MaskGenerator():
    def __init__(self, sam_checkpoint, args):
        self.args = args
        self.samcontroler = SamControler(sam_checkpoint, args.sam_model_type, args.device)
       
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image
    
# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

def get_frames_from_image(image_input, image_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """

    user_name = time.time()
    frames = [image_input] * 2  # hardcode: mimic a video with 2 frames
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    # initialize video_state
    image_state = {
        "user_name": user_name,
        "image_name": "output.png",
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": None
        }
    image_info = "Image Name: N/A,\nFPS: N/A,\nTotal Frames: {},\nImage Size:{}".format(len(frames), image_size)
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(image_state["origin_images"][0])
    return image_state, image_info, image_state["origin_images"][0], \
                        gr.update(visible=True, maximum=10, value=10), gr.update(visible=False, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=True), \
                        gr.update(visible=True)

# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    frames = []
    user_name = time.time()

    # extract Audio
    try:
        audio_path = video_input.replace(".mp4", "_audio.wav")
        ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=2, ar='44100').run(overwrite_output=True, quiet=True)
    except Exception as e:
        print(f"Audio extraction error: {str(e)}")
        audio_path = ""  # Set to "" if extraction fails
    
    # extract frames
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1]) 

    # [remove for local demo] resize if resolution too big
    # if image_size[0]>=1280 and image_size[0]>=1280:
    #     scale = 1080 / min(image_size)
    #     new_w = int(image_size[1] * scale)
    #     new_h = int(image_size[0] * scale)
    #     # update frames
    #     frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
    #     # update image_size
    #     image_size = (frames[0].shape[0],frames[0].shape[1]) 

    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps,
        "audio": audio_path
        }
    video_info = "Video Name: {},\nFPS: {},\nTotal Frames: {},\nImage Size:{}".format(video_state["video_name"], round(video_state["fps"], 0), len(frames), image_size)
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return video_state, video_info, video_state["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=False, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=True), \
                        gr.update(visible=True)

# get the select frame from gradio slider
def select_video_template(image_selection_slider, video_state, interactive_state):

    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state

def select_image_template(image_selection_slider, video_state, interactive_state):

    image_selection_slider = 0 # fixed for image
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider

    return video_state["painted_images"][track_pause_number_slider],interactive_state

# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    return painted_image, video_state, interactive_state

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    mask = video_state["masks"][video_state["select_frame_number"]]
    interactive_state["multi_mask"]["masks"].append(mask)
    interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    select_frame = show_mask(video_state, interactive_state, mask_dropdown)

    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]]

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    return template_frame, click_state

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    return interactive_state, gr.update(choices=[],value=[])

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    if video_state["origin_images"]:
        select_frame = video_state["origin_images"][video_state["select_frame_number"]]
        for i in range(len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            mask = interactive_state["multi_mask"]["masks"][mask_number]
            select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
        
        return select_frame

# image matting
def image_matting(video_state, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size, refine_iter):
    # * Move SAM to CPU and MatAnyone to GPU for matting
    move_sam_to_cpu()
    move_matting_to_gpu()
    
    matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
    foreground, alpha = matanyone(matanyone_processor, following_frames, template_mask*255, r_erode=erode_kernel_size, r_dilate=dilate_kernel_size, n_warmup=refine_iter)
    foreground_output = Image.fromarray(foreground[-1])
    alpha_output = Image.fromarray(alpha[-1][:,:,0])

    # * Move MatAnyone to CPU and SAM back to GPU after matting
    move_matting_to_cpu()
    move_sam_to_gpu()

    return foreground_output, alpha_output

# video matting
def video_matting(video_state, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size):
    # * Move SAM to CPU and MatAnyone to GPU for matting
    move_sam_to_cpu()
    move_matting_to_gpu()
    
    matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    audio_path = video_state["audio"]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
    foreground, alpha = matanyone(matanyone_processor, following_frames, template_mask*255, r_erode=erode_kernel_size, r_dilate=dilate_kernel_size)

    foreground_output = generate_video_from_frames(foreground, output_path="./results/{}_fg.mp4".format(video_state["video_name"]), fps=fps, audio_path=audio_path) # import video_input to name the output video
    alpha_output = generate_video_from_frames(alpha, output_path="./results/{}_alpha.mp4".format(video_state["video_name"]), fps=fps, gray2rgb=True, audio_path=audio_path) # import video_input to name the output video
    
    # * Move MatAnyone to CPU and SAM back to GPU after matting
    move_matting_to_cpu()
    move_sam_to_gpu()
    
    return foreground_output, alpha_output


def add_audio_to_video(video_path, audio_path, output_path):
    try:
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(audio_path)

        _ = (
            ffmpeg
            .output(video_input, audio_input, output_path, vcodec="copy", acodec="aac")
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        return output_path
    except ffmpeg.Error as e:
        print(f"FFmpeg error:\n{e.stderr.decode()}")
        return None


def generate_video_from_frames(frames, output_path, fps=30, gray2rgb=False, audio_path=""):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    _, h, w, _ = frames.shape
    if gray2rgb:
        frames = np.repeat(frames, 3, axis=3)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    video_temp_path = output_path.replace(".mp4", "_temp.mp4")
    
    # resize back to ensure input resolution
    imageio.mimwrite(video_temp_path, frames, fps=fps, quality=7, 
                     codec='libx264', ffmpeg_params=["-vf", f"scale={w}:{h}"])
    
    # add audio to video if audio path exists
    if audio_path != "" and os.path.exists(audio_path):
        output_path = add_audio_to_video(video_temp_path, audio_path, output_path)    
        os.remove(video_temp_path)
        return output_path
    else:
        return video_temp_path

# reset all states for a new input
def restart():
    return {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30
        }, {
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": args.mask_save,
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            "track_end_number": None,
        }, [[],[]], None, None, \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),\
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False, choices=[], value=[]), "", gr.update(visible=False)

# args, defined in track_anything.py
args = parse_augment()
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
checkpoint_folder = os.path.join('..', 'pretrained_models')

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], checkpoint_folder)
# initialize sams
model = MaskGenerator(sam_checkpoint, args)

# initialize matanyone
pretrain_model_url = "https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
ckpt_path = load_file_from_url(pretrain_model_url, checkpoint_folder)
matanyone_model = get_matanyone_model(ckpt_path, args.device)
# * Start with MatAnyone on CPU to save GPU memory for SAM
matanyone_model = matanyone_model.cpu().eval()
# matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)

# download test samples
test_sample_path = os.path.join('.', "test_sample/")
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/test-sample0-720p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/test-sample1-720p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/test-sample2-720p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/test-sample3-720p.mp4', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/test-sample0.jpg', test_sample_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/test-sample1.jpg', test_sample_path)

# download assets
assets_path = os.path.join('.', "assets/")
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/tutorial_single_target.mp4', assets_path)
load_file_from_url('https://github.com/pq-yang/MatAnyone/releases/download/media/tutorial_multi_targets.mp4', assets_path)

# documents
title = r"""<div class="multi-layer" align="center"><span>MatAnyone</span></div>
"""
description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/pq-yang/MatAnyone' target='_blank'><b>MatAnyone: Stable Video Matting with Consistent Memory Propagation</b></a>.<br>
🔥 MatAnyone is a practical human video matting framework supporting target assignment 🎯.<br>
🎪 Try to drop your video/image, assign the target masks with a few clicks, and get the the matting results 🤡!<br>

*Note: Due to the online GPU memory constraints, any input with too big resolution will be resized to 1080p.<br>*
🚀 <b> If you encounter any issue (e.g., frozen video output) or wish to run on higher resolution inputs, please consider <u>duplicating this space</u> or 
<u>launching the <a href='https://github.com/pq-yang/MatAnyone?tab=readme-ov-file#-interactive-demo' target='_blank'>demo</a> locally</u> following the GitHub instructions.</b>
"""
article = r"""<h3>
<b>If MatAnyone is helpful, please help to 🌟 the <a href='https://github.com/pq-yang/MatAnyone' target='_blank'>Github Repo</a>. Thanks!</b></h3>

---

📑 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@InProceedings{yang2025matanyone,
     title     = {{MatAnyone}: Stable Video Matting with Consistent Memory Propagation},
     author    = {Yang, Peiqing and Zhou, Shangchen and Zhao, Jixin and Tao, Qingyi and Loy, Chen Change},
     booktitle = {arXiv preprint arXiv:2501.14677},
     year      = {2025}
}
```
📝 **License**
<br>
This project is licensed under <a rel="license" href="https://github.com/pq-yang/MatAnyone/blob/main/LICENSE">S-Lab License 1.0</a>. 
Redistribution and use for non-commercial purposes should follow this license.
<br>
📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>peiqingyang99@outlook.com</b>.
<br>
👏 **Acknowledgement**
<br>
This project is built upon [Cutie](https://github.com/hkchengrex/Cutie), with the interactive demo adapted from [ProPainter](https://github.com/sczhou/ProPainter), leveraging segmentation capabilities from [Segment Anything](https://github.com/facebookresearch/segment-anything). Thanks for their awesome works!
"""

my_custom_css = """
.gradio-container {width: 85% !important; margin: 0 auto;}
.gr-monochrome-group {border-radius: 5px !important; border: revert-layer !important; border-width: 2px !important; color: black !important}
button {border-radius: 8px !important;}
.new_button {background-color: #171717 !important; color: #ffffff !important; border: none !important;}
.green_button {background-color: #4CAF50 !important; color: #ffffff !important; border: none !important;}
.new_button:hover {background-color: #4b4b4b !important;}
.green_button:hover {background-color: #77bd79 !important;}

.mask_button_group {gap: 10px !important;}
.video .wrap.svelte-lcpz3o {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: auto !important;
    max-height: 300px !important;
}
.video .wrap.svelte-lcpz3o > :first-child {
    height: auto !important;
    width: 100% !important;
    object-fit: contain !important;
}
.video .container.svelte-sxyn79 {
    display: none !important;
}
.margin_center {width: 50% !important; margin: auto !important;}
.jc_center {justify-content: center !important;}
.video-title {
    margin-bottom: 5px !important;
}
.custom-bg {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
    }

<style>
@import url('https://fonts.googleapis.com/css2?family=Sarpanch:wght@400;500;600;700;800;900&family=Sen:wght@400..800&family=Sixtyfour+Convergence&family=Stardos+Stencil:wght@400;700&display=swap');
body {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    background-color: #0d1117;
    font-family: Arial, sans-serif;
    font-size: 18px;
    }
.title-container {
    text-align: center;
    padding: 0;
    margin: 0;
    height: 5vh;
    width: 80vw;
    font-family: "Sarpanch", sans-serif;
    font-weight: 60;
}
#custom-markdown {
    font-family: "Roboto", sans-serif;
    font-size: 18px;
    color: #333333;
    font-weight: bold;
}
small {
    font-size: 60%;
}
</style>
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=my_custom_css) as demo:
    gr.HTML('''
        <div class="title-container">
            <h1 class="title is-2 publication-title"
                style="font-size:50px; font-family: 'Sarpanch', serif; 
                    background: linear-gradient(to right, #d231d8, #2dc464); 
                    display: inline-block; -webkit-background-clip: text; 
                    -webkit-text-fill-color: transparent;">
                MatAnyone
            </h1>
        </div>
    ''')

    gr.Markdown(description)

    with gr.Group(elem_classes="gr-monochrome-group", visible=True):
        with gr.Row():
            with gr.Accordion("📕 Video Tutorial (click to expand)", open=False, elem_classes="custom-bg"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Case 1: Single Target")
                        gr.Video(value="./assets/tutorial_single_target.mp4", elem_classes="video")

                    with gr.Column():
                        gr.Markdown("### Case 2: Multiple Targets")
                        gr.Video(value="./assets/tutorial_multi_targets.mp4", elem_classes="video")

    with gr.Tabs():
        with gr.TabItem("Video"):
            click_state = gr.State([[],[]])

            interactive_state = gr.State({
                "inference_times": 0,
                "negative_click_times" : 0,
                "positive_click_times": 0,
                "mask_save": args.mask_save,
                "multi_mask": {
                    "mask_names": [],
                    "masks": []
                },
                "track_end_number": None,
                }
            )

            video_state = gr.State(
                {
                "user_name": "",
                "video_name": "",
                "origin_images": None,
                "painted_images": None,
                "masks": None,
                "inpaint_masks": None,
                "logits": None,
                "select_frame_number": 0,
                "fps": 30,
                "audio": "",
                }
            )

            with gr.Group(elem_classes="gr-monochrome-group", visible=True):
                with gr.Row():
                    with gr.Accordion('MatAnyone Settings (click to expand)', open=False):
                        with gr.Row():
                            erode_kernel_size = gr.Slider(label='Erode Kernel Size',
                                                    minimum=0,
                                                    maximum=30,
                                                    step=1,
                                                    value=10,
                                                    info="Erosion on the added mask",
                                                    interactive=True)
                            dilate_kernel_size = gr.Slider(label='Dilate Kernel Size',
                                                    minimum=0,
                                                    maximum=30,
                                                    step=1,
                                                    value=10,
                                                    info="Dilation on the added mask",
                                                    interactive=True)

                        with gr.Row():
                            image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Start Frame", info="Choose the start frame for target assignment and video matting", visible=False)
                            track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Positive", "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                info="Click to add positive or negative point for target mask",
                                interactive=True,
                                visible=False,
                                min_width=100,
                                scale=1)
                            mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask Selection", info="Choose 1~all mask(s) added in Step 2", visible=False)
            
            gr.Markdown("---")

            with gr.Column():
                # input video
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2): 
                        gr.Markdown("## Step1: Upload video")
                    with gr.Column(scale=2): 
                        step2_title = gr.Markdown("## Step2: Add masks <small>(Several clicks then **`Add Mask`** <u>one by one</u>)</small>", visible=False)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):      
                        video_input = gr.Video(label="Input Video", elem_classes="video")
                        extract_frames_button = gr.Button(value="Load Video", interactive=True, elem_classes="new_button")
                    with gr.Column(scale=2):
                        video_info = gr.Textbox(label="Video Info", visible=False)
                        template_frame = gr.Image(label="Start Frame", type="pil",interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                        with gr.Row(equal_height=True, elem_classes="mask_button_group"):
                            clear_button_click = gr.Button(value="Clear Clicks", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            add_mask_button = gr.Button(value="Add Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            remove_mask_button = gr.Button(value="Remove Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100) # no use
                            matting_button = gr.Button(value="Video Matting", interactive=True, visible=False, elem_classes="green_button", min_width=100)
                
                gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #a566b4, #74a781);margin: 5px 0;">')

                # output video
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        foreground_video_output = gr.Video(label="Foreground Output", visible=False, elem_classes="video")
                        foreground_output_button = gr.Button(value="Foreground Output", visible=False, elem_classes="new_button")
                    with gr.Column(scale=2):
                        alpha_video_output = gr.Video(label="Alpha Output", visible=False, elem_classes="video")
                        alpha_output_button = gr.Button(value="Alpha Mask Output", visible=False, elem_classes="new_button")
                

            # first step: get the video information 
            extract_frames_button.click(
                fn=get_frames_from_video,
                inputs=[
                    video_input, video_state
                ],
                outputs=[video_state, video_info, template_frame,
                        image_selection_slider, track_pause_number_slider, point_prompt, clear_button_click, add_mask_button, matting_button, template_frame,
                        foreground_video_output, alpha_video_output, foreground_output_button, alpha_output_button, mask_dropdown, step2_title]
            )   

            # second step: select images from slider
            image_selection_slider.release(fn=select_video_template, 
                                        inputs=[image_selection_slider, video_state, interactive_state], 
                                        outputs=[template_frame, video_state, interactive_state], api_name="select_image")
            track_pause_number_slider.release(fn=get_end_number, 
                                        inputs=[track_pause_number_slider, video_state, interactive_state], 
                                        outputs=[template_frame, interactive_state], api_name="end_image")
            
            # click select image to get mask using sam
            template_frame.select(
                fn=sam_refine,
                inputs=[video_state, point_prompt, click_state, interactive_state],
                outputs=[template_frame, video_state, interactive_state]
            )

            # add different mask
            add_mask_button.click(
                fn=add_multi_mask,
                inputs=[video_state, interactive_state, mask_dropdown],
                outputs=[interactive_state, mask_dropdown, template_frame, click_state]
            )

            remove_mask_button.click(
                fn=remove_multi_mask,
                inputs=[interactive_state, mask_dropdown],
                outputs=[interactive_state, mask_dropdown]
            )

            # video matting
            matting_button.click(
                fn=video_matting,
                inputs=[video_state, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size],
                outputs=[foreground_video_output, alpha_video_output]
            )

            # click to get mask
            mask_dropdown.change(
                fn=show_mask,
                inputs=[video_state, interactive_state, mask_dropdown],
                outputs=[template_frame]
            )
            
            # clear input
            video_input.change(
                fn=restart,
                inputs=[],
                outputs=[ 
                    video_state,
                    interactive_state,
                    click_state,
                    foreground_video_output, alpha_video_output,
                    template_frame,
                    image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
                    add_mask_button, matting_button, template_frame, foreground_video_output, alpha_video_output, remove_mask_button, foreground_output_button, alpha_output_button, mask_dropdown, video_info, step2_title
                ],
                queue=False,
                show_progress=False)
            
            video_input.clear(
                fn=restart,
                inputs=[],
                outputs=[ 
                    video_state,
                    interactive_state,
                    click_state,
                    foreground_video_output, alpha_video_output,
                    template_frame,
                    image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
                    add_mask_button, matting_button, template_frame, foreground_video_output, alpha_video_output, remove_mask_button, foreground_output_button, alpha_output_button, mask_dropdown, video_info, step2_title
                ],
                queue=False,
                show_progress=False)
            
            # points clear
            clear_button_click.click(
                fn = clear_click,
                inputs = [video_state, click_state,],
                outputs = [template_frame,click_state],
            )

            # set example
            gr.Markdown("---")
            gr.Markdown("## Examples")
            gr.Examples(
                examples=[os.path.join(os.path.dirname(__file__), "./test_sample/", test_sample) for test_sample in ["test-sample0-720p.mp4", "test-sample1-720p.mp4", "test-sample2-720p.mp4", "test-sample3-720p.mp4"]],
                inputs=[video_input],
            )

        with gr.TabItem("Image"):
            click_state = gr.State([[],[]])

            interactive_state = gr.State({
                "inference_times": 0,
                "negative_click_times" : 0,
                "positive_click_times": 0,
                "mask_save": args.mask_save,
                "multi_mask": {
                    "mask_names": [],
                    "masks": []
                },
                "track_end_number": None,
                }
            )

            image_state = gr.State(
                {
                "user_name": "",
                "image_name": "",
                "origin_images": None,
                "painted_images": None,
                "masks": None,
                "inpaint_masks": None,
                "logits": None,
                "select_frame_number": 0,
                "fps": 30
                }
            )

            with gr.Group(elem_classes="gr-monochrome-group", visible=True):
                with gr.Row():
                    with gr.Accordion('MatAnyone Settings (click to expand)', open=False):
                        with gr.Row():
                            erode_kernel_size = gr.Slider(label='Erode Kernel Size',
                                                    minimum=0,
                                                    maximum=30,
                                                    step=1,
                                                    value=10,
                                                    info="Erosion on the added mask",
                                                    interactive=True)
                            dilate_kernel_size = gr.Slider(label='Dilate Kernel Size',
                                                    minimum=0,
                                                    maximum=30,
                                                    step=1,
                                                    value=10,
                                                    info="Dilation on the added mask",
                                                    interactive=True)
                            
                        with gr.Row():
                            image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Num of Refinement Iterations", info="More iterations → More details & More time", visible=False)
                            track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Positive", "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                info="Click to add positive or negative point for target mask",
                                interactive=True,
                                visible=False,
                                min_width=100,
                                scale=1)
                            mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask Selection", info="Choose 1~all mask(s) added in Step 2", visible=False)
            
            gr.Markdown("---")

            with gr.Column():
                # input image
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2): 
                        gr.Markdown("## Step1: Upload image")
                    with gr.Column(scale=2): 
                        step2_title = gr.Markdown("## Step2: Add masks <small>(Several clicks then **`Add Mask`** <u>one by one</u>)</small>", visible=False)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):      
                        image_input = gr.Image(label="Input Image", elem_classes="image")
                        extract_frames_button = gr.Button(value="Load Image", interactive=True, elem_classes="new_button")
                    with gr.Column(scale=2):
                        image_info = gr.Textbox(label="Image Info", visible=False)
                        template_frame = gr.Image(type="pil", label="Start Frame", interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                        with gr.Row(equal_height=True, elem_classes="mask_button_group"):
                            clear_button_click = gr.Button(value="Clear Clicks", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            add_mask_button = gr.Button(value="Add Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            remove_mask_button = gr.Button(value="Remove Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                            matting_button = gr.Button(value="Image Matting", interactive=True, visible=False, elem_classes="green_button", min_width=100)

                gr.HTML('<hr style="border: none; height: 1.5px; background: linear-gradient(to right, #a566b4, #74a781);margin: 5px 0;">')

                # output image
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        foreground_image_output = gr.Image(type="pil", label="Foreground Output", visible=False, elem_classes="image")
                        foreground_output_button = gr.Button(value="Foreground Output", visible=False, elem_classes="new_button")
                    with gr.Column(scale=2):
                        alpha_image_output = gr.Image(type="pil", label="Alpha Output", visible=False, elem_classes="image")
                        alpha_output_button = gr.Button(value="Alpha Mask Output", visible=False, elem_classes="new_button")

            # first step: get the image information 
            extract_frames_button.click(
                fn=get_frames_from_image,
                inputs=[
                    image_input, image_state
                ],
                outputs=[image_state, image_info, template_frame,
                        image_selection_slider, track_pause_number_slider,point_prompt, clear_button_click, add_mask_button, matting_button, template_frame,
                        foreground_image_output, alpha_image_output, foreground_output_button, alpha_output_button, mask_dropdown, step2_title]
            )   

            # second step: select images from slider
            image_selection_slider.release(fn=select_image_template, 
                                        inputs=[image_selection_slider, image_state, interactive_state], 
                                        outputs=[template_frame, image_state, interactive_state], api_name="select_image")
            track_pause_number_slider.release(fn=get_end_number, 
                                        inputs=[track_pause_number_slider, image_state, interactive_state], 
                                        outputs=[template_frame, interactive_state], api_name="end_image")
            
            # click select image to get mask using sam
            template_frame.select(
                fn=sam_refine,
                inputs=[image_state, point_prompt, click_state, interactive_state],
                outputs=[template_frame, image_state, interactive_state]
            )

            # add different mask
            add_mask_button.click(
                fn=add_multi_mask,
                inputs=[image_state, interactive_state, mask_dropdown],
                outputs=[interactive_state, mask_dropdown, template_frame, click_state]
            )

            remove_mask_button.click(
                fn=remove_multi_mask,
                inputs=[interactive_state, mask_dropdown],
                outputs=[interactive_state, mask_dropdown]
            )

            # image matting
            matting_button.click(
                fn=image_matting,
                inputs=[image_state, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size, image_selection_slider],
                outputs=[foreground_image_output, alpha_image_output]
            )

            # click to get mask
            mask_dropdown.change(
                fn=show_mask,
                inputs=[image_state, interactive_state, mask_dropdown],
                outputs=[template_frame]
            )
            
            # clear input
            image_input.change(
                fn=restart,
                inputs=[],
                outputs=[ 
                    image_state,
                    interactive_state,
                    click_state,
                    foreground_image_output, alpha_image_output,
                    template_frame,
                    image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
                    add_mask_button, matting_button, template_frame, foreground_image_output, alpha_image_output, remove_mask_button, foreground_output_button, alpha_output_button, mask_dropdown, image_info, step2_title
                ],
                queue=False,
                show_progress=False)
            
            image_input.clear(
                fn=restart,
                inputs=[],
                outputs=[ 
                    image_state,
                    interactive_state,
                    click_state,
                    foreground_image_output, alpha_image_output,
                    template_frame,
                    image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
                    add_mask_button, matting_button, template_frame, foreground_image_output, alpha_image_output, remove_mask_button, foreground_output_button, alpha_output_button, mask_dropdown, image_info, step2_title
                ],
                queue=False,
                show_progress=False)
            
            # points clear
            clear_button_click.click(
                fn = clear_click,
                inputs = [image_state, click_state,],
                outputs = [template_frame,click_state],
            )

            # set example
            gr.Markdown("---")
            gr.Markdown("## Examples")
            gr.Examples(
                examples=[os.path.join(os.path.dirname(__file__), "./test_sample/", test_sample) for test_sample in ["test-sample0.jpg", "test-sample1.jpg"]],
                inputs=[image_input],
            )

    gr.Markdown(article)

demo.queue()
demo.launch(debug=True)