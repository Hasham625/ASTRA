#!/bin/bash
#
# stop_train.sh — Cleanly stops the ASTrA training process.
# Invoked by cron at 6pm daily. Uses SIGTERM first so the process
# can save state, then escalates to SIGKILL after 60 seconds.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

LOG_FILE="$SCRIPT_DIR/schedule_train.log"
PID_FILE="$SCRIPT_DIR/training.pid"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

if [ ! -f "$PID_FILE" ]; then
    log "No PID file found. Nothing to stop."
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    log "PID $PID not running. Cleaning up PID file."
    rm -f "$PID_FILE"
    exit 0
fi

log "Sending SIGTERM to training process (PID $PID)."
kill -TERM "$PID"

# Wait up to 60 seconds for graceful shutdown
for i in {1..60}; do
    sleep 1
    if ! kill -0 "$PID" 2>/dev/null; then
        log "Training stopped gracefully after ${i}s."
        rm -f "$PID_FILE"
        exit 0
    fi
done

log "Process still running after 60s. Sending SIGKILL."
kill -KILL "$PID" 2>/dev/null
rm -f "$PID_FILE"
log "Training force-killed."
