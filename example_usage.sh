#!/bin/bash

# Example 1: Basic training with default settings
python main.py

# Example 2: Custom configuration with 8 clients (2 tasks x 4 clients each)
python main.py \
    --model_name roberta-large \
    --tasks sst2 sst2 qnli qnli \
    --global_rounds 20 \
    --local_epochs 2 \
    --warmup_rounds 3 \
    --max_clusters 2 \
    --lr 3e-3 \
    --batch_size 128

# Example 3: Larger setup with 16 clients (4 tasks x 4 clients each)
python main.py \
    --model_name roberta-large \
    --tasks sst2 sst2 sst2 sst2 qnli qnli qnli qnli mrpc mrpc mrpc mrpc qqp qqp qqp qqp \
    --global_rounds 25 \
    --local_epochs 2 \
    --warmup_rounds 5 \
    --max_clusters 4 \
    --train_samples 1000 \
    --test_samples 200 \
    --output_dir ./output

# Example 4: Quick test with fewer samples
python main.py \
    --tasks sst2 qnli \
    --global_rounds 5 \
    --warmup_rounds 2 \
    --train_samples 100 \
    --test_samples 50 \
    --max_clusters 2
