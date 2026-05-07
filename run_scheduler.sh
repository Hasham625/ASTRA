#!/bin/bash
#
# run_scheduler.sh — Simple user-mode scheduler alternative to cron.
# Runs forever, checking every 5 minutes whether to start/stop training.
# Launch with: nohup ./run_scheduler.sh > scheduler.log 2>&1 &
# Stop with:   pkill -f run_scheduler.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

SCHEDULER_LOG="$SCRIPT_DIR/scheduler.log"
START_HOUR=7
END_HOUR=18

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$SCHEDULER_LOG"
}

log "Scheduler started (window: $START_HOUR:00 - $END_HOUR:00)."

while true; do
    CURRENT_HOUR=$(date +%H)
    CURRENT_HOUR=${CURRENT_HOUR#0}

    if [ "$CURRENT_HOUR" -ge "$START_HOUR" ] && [ "$CURRENT_HOUR" -lt "$END_HOUR" ]; then
        # Inside window: ensure training is running
        bash "$SCRIPT_DIR/schedule_train.sh"
    else
        # Outside window: ensure training is stopped
        if [ -f "$SCRIPT_DIR/training.pid" ]; then
            bash "$SCRIPT_DIR/stop_train.sh"
        fi
    fi

    sleep 300   # check every 5 minutes
done
