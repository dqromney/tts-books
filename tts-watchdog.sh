#!/usr/bin/env bash
# tts-watchdog.sh — Monitors TTS memory and restarts before OOM
# Run via cron every 5 minutes: */5 * * * * ~/bin/tts-watchdog.sh
set -e

LOG="/home/dqromney/tts_output/watchdog.log"
RSS_LIMIT_GB=${RSS_LIMIT_GB:-50}  # Kill if RSS exceeds this (GB)
HOME_DIR="/home/dqromney"
XAUTHORITY="/run/user/1000/gdm/Xauthority"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

# Find the TTS Python process (tts_book_gui.py)
PID=$(pgrep -f "tts_book_gui.py" | head -1)

if [ -z "$PID" ]; then
    # No TTS running — check if there's a queue with pending items to resume
    QUEUE="$HOME_DIR/bin/tts-book.queue.json"
    if [ -f "$QUEUE" ]; then
        PENDING=$(python3 -c "
import json
with open('$QUEUE') as f:
    items = json.load(f).get('items', [])
print(len([i for i in items if i.get('status') == 'pending']))" 2>/dev/null || echo 0)

        if [ "$PENDING" -gt 0 ]; then
            # Guard against fail-loop: check when we last tried a restart
            LAST_RESTART=0
            if [ -f "$HOME_DIR/tts_output/.last_restart" ]; then
                LAST_RESTART=$(cat "$HOME_DIR/tts_output/.last_restart")
            fi
            NOW=$(date +%s)
            DIFF=$((NOW - LAST_RESTART))
            if [ "$DIFF" -lt 1800 ]; then  # 30 min cooldown
                # TTS died but we restarted recently — skip to avoid loop
                exit 0
            fi
            log "No TTS running but $PENDING pending queue items — restarting with --auto-resume"
            export DISPLAY=:1
            export XAUTHORITY="$XAUTHORITY"
            date +%s > "$HOME_DIR/tts_output/.last_restart"
            nohup "$HOME_DIR/bin/tts-book.sh" --auto-resume > /dev/null 2>&1 &
            log "Restarted PID: $!"
        fi
    fi
    exit 0
fi

# Get RSS in KB, convert to GB
RSS_KB=$(awk '/^VmRSS:/ {print $2}' "/proc/$PID/status" 2>/dev/null || echo 0)
RSS_GB=$((RSS_KB / 1024 / 1024))

log "TTS PID $PID: RSS=${RSS_GB}GB (limit=${RSS_LIMIT_GB}GB)"

if [ "$RSS_GB" -ge "$RSS_LIMIT_GB" ]; then
    log "ALERT: RSS ${RSS_GB}GB >= limit ${RSS_LIMIT_GB}GB — killing PID $PID"
    kill "$PID" 2>/dev/null || true
    sleep 3
    # Force kill if still alive
    kill -9 "$PID" 2>/dev/null || true
    sleep 2
    
    # Restart with auto-resume
    export DISPLAY=:1
    export XAUTHORITY="$XAUTHORITY"
    nohup "$HOME_DIR/bin/tts-book.sh" --auto-resume > /dev/null 2>&1 &
    log "Restarted PID: $!"
fi
