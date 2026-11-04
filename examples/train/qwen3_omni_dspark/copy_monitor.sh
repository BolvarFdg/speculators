#!/bin/bash
# File-count monitor + ETA for start_copy.sh (works even with the silent
# fan-out workers). Run in a separate terminal; Ctrl-C to stop.
#
# The source count is taken once (local SSD, fast); the destination count
# walks the NFS tree (takes minutes at 1M files -- that is why the default
# interval is 15 min).

set -uo pipefail

# ================= Configuration (env-overridable) =================
SRC="${SRC:-/ssd/audios}"                 # source root (same as start_copy.sh)
DST="${DST:-/nfs/target/audios}"          # destination root (same as start_copy.sh)
INTERVAL="${INTERVAL:-3600}"               # seconds between samples. Keep >=
                                           # 3600 on slow NFS: each sample
                                           # walks every destination file
                                           # (1M metadata round trips) and
                                           # competes with the copy for the
                                           # server's ops/sec -- a long
                                           # interval makes that negligible.
# =====================================================================

count_wav() { find "$1" -name '*.wav' 2>/dev/null | wc -l; }

echo "counting source (local, fast)..."
TOTAL=$(count_wav "$SRC")
echo "total wav: $TOTAL"
[[ "$TOTAL" -gt 0 ]] || { echo "ERROR: no wav under SRC=$SRC"; exit 1; }

echo "first destination count (NFS walk, may take minutes)..."
PREV=$(count_wav "$DST")
PREV_T=$SECONDS

while true; do
    sleep "$INTERVAL"
    CUR=$(count_wav "$DST")
    NOW=$SECONDS
    ELAPSED=$(( NOW - PREV_T ))
    PCT=$(( CUR * 100 / TOTAL ))
    if [[ "$ELAPSED" -gt 0 ]]; then
        RATE=$(( (CUR - PREV) * 60 / ELAPED ))   # files per minute
    else
        RATE=0
    fi
    LEFT=$(( TOTAL - CUR ))
    if [[ "$RATE" -gt 0 ]]; then
        ETA_MIN=$(( LEFT / RATE ))
        printf '%s  %d/%d (%d%%)  %d/min  ETA ~%dh%02dm\n' \
            "$(date '+%m-%d %H:%M')" "$CUR" "$TOTAL" "$PCT" "$RATE" \
            "$((ETA_MIN / 60))" "$((ETA_MIN % 60))"
    else
        # Zero growth with the copy still running usually means the
        # workers are in the NFS stat-comparison phase for already-done
        # units -- not a stall. Check worker count to tell them apart:
        printf '%s  %d/%d (%d%%)  rate=0  (workers: %s)\n' \
            "$(date '+%m-%d %H:%M')" "$CUR" "$TOTAL" "$PCT" \
            "$(pgrep -fc 'rsync -a' || echo 0)"
    fi
    PREV=$CUR
    PREV_T=$NOW
    if [[ "$CUR" -ge "$TOTAL" ]]; then
        echo "all files present -- script is now in the catch-up/verify"
        echo "phase (NFS stat pass over everything, can take 30-60+ min);"
        echo "watch for 'DONE' in status.txt or the main pid exiting."
        break
    fi
done
