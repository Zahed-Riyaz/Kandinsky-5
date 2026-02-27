#!/bin/bash

# path to folder with pairs image-caption
IMAGES_CAPTIONS_DIR="./captions_dir"

# for training t2i
SAVE_LATENTS_DIR_IMAGE="cache/latents_image"

python encode/encode_images.py \
    --images_captions_dir="$IMAGES_CAPTIONS_DIR" \
    --save_latents_dir="$SAVE_LATENTS_DIR_IMAGE" \
    --config="configs/model/flux.yaml" \
    --device=0


# # for training t2v even on images
SAVE_LATENTS_DIR_VIDEO="cache/latents_video"

python encode/encode_images.py \
    --images_captions_dir="$IMAGES_CAPTIONS_DIR" \
    --save_latents_dir="$SAVE_LATENTS_DIR_VIDEO" \
    --config="configs/model/hunyuan.yaml" \
    --device=0 


SAVE_TEXT_EMBEDS_DIR="cache/text_embeds"

python encode/encode_captions.py \
    --images_captions_dir="$IMAGES_CAPTIONS_DIR" \
    --save_text_embeds_dir="$SAVE_TEXT_EMBEDS_DIR" \
    --config='configs/model/clip_qwen7b.yaml' \
    --device=0 
