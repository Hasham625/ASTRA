#!/bin/bash
#
# schedule_train.sh — Starts ASTrA training within the 7am–6pm window.
# Exits immediately if outside the window. Automatically resumes from the
# latest checkpoint if one exists.
#
# Intended to be invoked by a cron job every 5 minutes. If training is
# already running, the new invocation exits silently.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# ---- Time window (24-hour format) ----
START_HOUR=6   # TEMPORARY for testing (revert to 7 after)
END_HOUR=18    # 6pm

LOG_FILE="$SCRIPT_DIR/schedule_train.log"
PID_FILE="$SCRIPT_DIR/training.pid"
CHECKPOINT_FILE="$SCRIPT_DIR/checkpoints/ASTrA_ViT_Training/model_both.pt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# ---- Check if outside training window ----
CURRENT_HOUR=$(date +%H)
CURRENT_HOUR=${CURRENT_HOUR#0}  # strip leading zero for arithmetic

if [ "$CURRENT_HOUR" -lt "$START_HOUR" ] || [ "$CURRENT_HOUR" -ge "$END_HOUR" ]; then
    log "Outside training window ($START_HOUR:00 - $END_HOUR:00). Exiting."
    exit 0
fi

# ---- Check if training is already running ----
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        log "Training already running with PID $OLD_PID. Exiting."
        exit 0
    else
        log "Stale PID file found. Cleaning up."
        rm -f "$PID_FILE"
    fi
fi

# ---- Build resume flag if checkpoint exists ----
RESUME_FLAG=""
if [ -f "$CHECKPOINT_FILE" ]; then
    RESUME_FLAG="--resume"
    log "Checkpoint found. Resuming from last epoch."
else
    log "No checkpoint found. Starting from scratch."
fi

# ---- Launch training ----
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled         # fully bypass wandb (no login needed)
export WANDB_SILENT=true            # suppress wandb output

log "Starting training (batch=192, window $START_HOUR:00-$END_HOUR:00)."

cd "$SCRIPT_DIR"
nohup python3 -u "$SCRIPT_DIR/train_simCLR_adaptive_update.py" \
    --experiment ASTrA_ViT_Training \
    --batch_size 192 \
    --interval_num 10 \
    --r1 0 \
    --sim_weight 0.5 \
    --ACL_DS \
    --gpu_ids "0" \
    --dataset "cifar10" \
    --policy_model_lr 0.1 \
    --img_size 32 \
    --patch_size 4 \
    --epochs 1000 \
    $RESUME_FLAG \
    >> "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
log "Training launched with PID $TRAIN_PID."
