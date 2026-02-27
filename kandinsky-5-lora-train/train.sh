#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/lora_video.yaml"

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv-endpoint="127.0.0.1:55154" \
    --rdzv-backend="static" \
    main.py \
        --config="$CONFIG"