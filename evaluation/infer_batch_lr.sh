#!/bin/bash

input_folder="./data/YouTubeMatte/youtubematte_512x288"
mask_folder="./data/YouTubeMatte_first_frame_seg_mask/youtubematte_512x288"

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
                  --output_path "./data/results/youtubematte_512x288/${subfolder}" \
                  --warmup 1 \
                  --erode_kernel 4 \
                  --dilate_kernel 4 \
                  --save_image
        fi
      fi
    fi
  done
done