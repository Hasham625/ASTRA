#!/bin/bash

# Get the directory of the run.sh script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (two levels up from the script directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Reduce memory fragmentation on small-VRAM GPUs (e.g. A2000 12GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "$SCRIPT_DIR/train_simCLR_adaptive_update.py" \
    --experiment ASTrA_ViT_Training \
    --batch_size 64 \
    --interval_num 10 \
    --r1 0 \
    --sim_weight 0.5 \
    --ACL_DS \
    --gpu_ids "0" \
    --dataset "cifar10" \
    --policy_model_lr 0.1 \
    --img_size 32 \
    --patch_size 4 \
    --epochs 5 \
    
    
