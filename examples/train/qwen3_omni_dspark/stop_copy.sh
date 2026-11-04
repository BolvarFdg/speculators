#!/bin/bash
# Part 1/2: stop the running copy (main script first, then the rsync workers).
# Order matters: killing only the rsync workers would make the original
# copy_audios.sh record FAILs and start its single-stream catch-up pass.

set -uo pipefail

PID_FILE="${PID_FILE:-copy.pid}"

if [[ -f "$PID_FILE" ]]; then
    MAIN_PID=$(cat "$PID_FILE")
    if kill "$MAIN_PID" 2>/dev/null; then
        echo "stopped main script (pid $MAIN_PID)"
    else
        echo "main script (pid $MAIN_PID) not running"
    fi
else
    echo "no $PID_FILE found; killing any copy_audios.sh directly"
    pkill -f copy_audios.sh 2>/dev/null
fi

pkill -f 'rsync -a' 2>/dev/null \
    && echo "killed rsync workers" \
    || echo "no rsync workers running"

sleep 2
LEFT=$(pgrep -f 'rsync -a|copy_audios.sh' | wc -l)
echo "remaining related processes: $LEFT"
[[ "$LEFT" -eq 0 ]] && echo "all stopped, ready for start script" \
                     || echo "WARNING: some processes survived; check: ps aux | grep rsync"
