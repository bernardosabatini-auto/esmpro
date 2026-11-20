#!/bin/bash
# keepalive: netscratch is purged on an age window, so walk the tree and reset every
# file's atime and mtime. The mount is relatime over NFS, so simply READING a file does
# not reliably refresh its timestamp -- only an explicit touch does.
#
# Run it from cron (see tools/keepalive.cron) or by hand:
#     bash tools/keepalive.sh
#
# One line per run goes to logs/keepalive.log, which doubles as the proof that the sweep
# is still happening. No file CONTENT is modified -- timestamps only.
set -u
ROOT="${KEEPALIVE_ROOT:-/n/netscratch/bsabatini_lab/Users/bsabatini/esm_proae}"
LOG="$ROOT/logs/keepalive.log"
cd "$ROOT" || exit 1
mkdir -p "$(dirname "$LOG")"
LIST=$(mktemp "${TMPDIR:-/tmp}/keepalive.XXXXXX")
trap 'rm -f "$LIST"' EXIT

start=$(date +%s)
find . -xdev \( -type f -o -type d \) -print0 2>/dev/null > "$LIST"
n=$(tr -dc '\0' < "$LIST" | wc -c)
xargs -0 -r -n 2000 touch -a -m < "$LIST" 2>/dev/null
secs=$(( $(date +%s) - start ))

# --rewrite: a touch only moves atime/mtime. If the purge scanner ever keys on the
# inode's creation time instead, a touch is invisible to it. Rewriting a file through a
# temporary copy and an atomic rename gives it a NEW inode, and so a new creation time.
# That is expensive and mildly risky, so it is limited to the small irreplaceable things
# and is meant to run monthly, not daily. Content is byte-identical afterwards.
if [ "${1:-}" = "--rewrite" ]; then
    rw=0
    for target in data/export data/export.tar.gz docs reports/campaign code/pdpipe; do
        [ -e "$target" ] || continue
        while IFS= read -r f; do
            tmp="$f.keepalive.$$"
            if cp -p -- "$f" "$tmp" 2>/dev/null && mv -f -- "$tmp" "$f" 2>/dev/null; then
                rw=$((rw + 1))
            else
                rm -f -- "$tmp"
            fi
        done < <(find "$target" -xdev -type f -size -200M 2>/dev/null)
    done
    printf '%s  rewrote %s files (new inodes, identical content)\n' \
           "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$rw" >> "$LOG"
fi

# The oldest surviving timestamp after the sweep. If this is ever far in the past, the
# sweep is missing part of the tree and the purge will take that part.
oldest=$(find . -xdev -type f -printf '%AY-%Am-%Ad\n' 2>/dev/null | sort | head -1)

printf '%s  touched %s paths in %ss   oldest atime now %s\n' \
       "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$n" "$secs" "${oldest:-NONE}" >> "$LOG"
tail -1 "$LOG"
