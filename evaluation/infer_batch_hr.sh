#!/bin/bash

input_folder="./data/YouTubeMatte/youtubematte_1920x1080"
mask_folder="./data/YouTubeMatte_first_frame_seg_mask/youtubematte_1920x1080"

for subfolder in "youtubematte_motion" "youtubematte_static"; do
  subfolder_path="${input_folder}/${subfolder}"
  
  echo "Processing subfolder: ${subfolder}"
  
  for video_folder in "${subfolder_path}"/*; do
    if [ -d "${video_folder}" ]; then
      video_id=$(basename "${video_folder}")

      mask_file="${mask_folder}/${video_id}.png"
      if [ -f "${mask_file}" ]; then

        input_frames_folder="${video_folder}/har"
        if [ -d "${input_frames_folder}" ]; then
          echo "Processing video: ${video_id} from ${subfolder}"
          
          python evaluation/inference_matanyone_yt.py \
                  --input_path "${input_frames_folder}" \
                  --mask_path "${mask_file}" \
                  --output_path "./data/results/youtubematte_1920x1080/${subfolder}" \
                  --warmup 10 \
                  --erode_kernel 15 \
                  --dilate_kernel 15 \
                  --save_image
        fi
      fi
    fi
  done
done