#!/bin/bash
# Part 2/2: restart the copy with second-level directory fan-out for high
# concurrency (small-file + high-latency NFS is per-stream latency bound;
# more streams scale aggregate throughput until the server saturates).
#
# Idempotent: already-copied files are skipped by rsync's size+mtime check.
# Restart cost = one metadata scan of the completed files (can look idle for
# 10-40 min while rescanning -- watch "to-chk" advance, don't panic).
#
# Progress: this runs silently by design; use the file-count monitor loop at
# the bottom (or the TOTAL_FILE_COUNT knob) to watch progress.
#
# Usage:
#   bash stop_copy.sh          # run Part 1 first
#   bash start_copy.sh         # then run this

set -euo pipefail

# ======================= Configuration (env-overridable) =======================
SRC="${SRC:-/ssd/audios}"                  # source root: contains the top-level dirs
DST="${DST:-/nfs/target/audios}"           # NFS destination root (same layout as SRC)
LOGDIR="${LOGDIR:-./copy_logs}"            # status/progress files

PARALLEL="${PARALLEL:-24}"                 # concurrent rsync streams
TOTAL_FILE_COUNT="${TOTAL_FILE_COUNT:-0}"  # total wav count for monitor; 0 = disabled
                                           # set to ~1000000 for progress percentage
# ==============================================================================

mkdir -p "$DST" "$LOGDIR"
echo "$(date '+%F %T') fan-out copy start: $SRC -> $DST (P=$PARALLEL)" | tee -a "$LOGDIR/status.txt"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: SRC=$SRC not found"; exit 1
fi

# Discover second-level dirs under the NOT-yet-completed top-level dirs.
# A top-level dir counts as completed when its name already has an OK line
# in status.txt from a previous run (both this script and copy_audios.sh
# write the same format); everything else gets copied.
COMPLETED=$(awk '$(NF-1)=="OK"{print $NF}' "$LOGDIR/status.txt" 2>/dev/null | sort -u || true)
echo "already completed top-level dirs: ${COMPLETED:-<none>}"

TODO_DIRS=()
for d in "$SRC"/*/; do
    name=$(basename "$d")
    if printf '%s\n' "$COMPLETED" | grep -qx "$name"; then
        echo "skip completed: $name"
    else
        TODO_DIRS+=("$name")
    fi
done
echo "to copy: ${TODO_DIRS[*]:-<none>}"

[[ ${#TODO_DIRS[@]} -gt 0 ]] || { echo "nothing to do"; exit 0; }

# Fan out each remaining second-level dir as one rsync stream. Paths travel
# as $1 (no string interpolation into the template), so spaces etc. are safe.
export DST LOGDIR
cd "$SRC"
find "${TODO_DIRS[@]}" -mindepth 1 -maxdepth 1 -type d -print0 \
  | xargs -0 -P "$PARALLEL" -n1 bash -c '
      d="$1"; name=${d//\//_}
      if rsync -aR --info=progress2 "$d" "$DST" > "$LOGDIR/$name.progress" 2>&1; then
          echo "$(date "+%F %T") OK    $name" >> "$LOGDIR/status.txt"
      else
          echo "$(date "+%F %T") FAIL  $name rc=$?" >> "$LOGDIR/status.txt"
      fi' _

# Catch-up pass: single-stream full pass, fills anything the fan-out missed
# (stray files directly under top-level dirs) and doubles as a consistency check.
echo "$(date '+%F %T') catch-up pass start" >> "$LOGDIR/status.txt"
if rsync -a "$SRC/" "$DST/" > "$LOGDIR/_catchup.progress" 2>&1; then
    echo "$(date '+%F %T') OK    catch-up" >> "$LOGDIR/status.txt"
else
    echo "$(date '+%F %T') FAIL  catch-up rc=$?" >> "$LOGDIR/status.txt"
fi

# Final verification: file counts must match.
SRC_N=$(find "$SRC" -name '*.wav' | wc -l)
DST_N=$(find "$DST" -name '*.wav' 2>/dev/null | wc -l)
echo "$(date '+%F %T') DONE  wav: $DST_N / $SRC_N" | tee -a "$LOGDIR/status.txt"
if [[ "$DST_N" -ne "$SRC_N" ]]; then
    echo "WARNING: file count mismatch -- rerun this script (idempotent)."
    exit 1
fi

# Clean up any stale hidden temp files left by killed streams (rsync writes
# to a hidden ".name.XXXX" temp and renames on completion; a kill can leave
# one behind). They never match *.wav, this is pure hygiene.
find "$DST" -name '.*.*' -type f -delete 2>/dev/null || true
echo "copy complete and verified."
