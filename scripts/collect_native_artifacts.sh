#!/usr/bin/env bash
# Collect native dual-boot test artifacts into one folder + tarball.
# Usage (on the Ubuntu box, any cwd):
#   bash scripts/collect_native_artifacts.sh [dest_dir]
# Default dest: ~/native_artifacts_<timestamp>  →  ~/native_artifacts_<timestamp>.tar.gz
# Copy that ONE file to the Windows side however you like.
#
# Alternative one-step (no tarball): mount the Windows partition from Ubuntu
# and copy straight into the repo:
#   lsblk -f                          # find the big NTFS partition (Windows C:)
#   sudo mkdir -p /mnt/win && sudo mount /dev/nvme0n1pX /mnt/win
#   cp ~/native_artifacts_*.tar.gz /mnt/win/Users/Lenovo/Desktop/Personal/RoutingComm/LinuxTests/
# (If the NTFS mount refuses or is read-only: Windows Fast Startup is holding
#  it — run `powercfg /h off` in Windows once, then reboot into Ubuntu.)

REPO="${REPO:-$HOME/Desktop/RoutingComm}"
OUT="${1:-$HOME/native_artifacts_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

echo "== repo CSVs =="
for f in ubuntu_tier1.csv ubuntu_tier3_run1.csv; do
  src="$REPO/data/raw/$f"
  if [ -f "$src" ]; then
    cp "$src" "$OUT/"
    printf '  %-24s %s lines (incl. header)\n' "$f" "$(wc -l < "$src")"
  else
    echo "  MISSING: $src"
  fi
done

echo
echo "== ros launch logs (5 newest sessions in ~/.ros/log) =="
n=0
for d in $(ls -1t "$HOME/.ros/log" 2>/dev/null | head -5); do
  log="$HOME/.ros/log/$d/launch.log"
  if [ -f "$log" ]; then
    cp "$log" "$OUT/launch_${d}.log"
    errs=$(grep -icE "traceback|unhandled|process has died" "$log" || true)
    echo "  ${d}: error-flagged lines = ${errs}"
    n=$((n + 1))
  fi
done
[ "$n" -eq 0 ] && echo "  (no launch.log files found)"

echo
echo "== rosbag (only if a re-record ever succeeded) =="
for bag in "$REPO/ubuntu_tier3_run1" "$HOME/ubuntu_tier3_run1"; do
  if [ -d "$bag" ]; then
    tar -czf "$OUT/$(basename "$bag").tar.gz" -C "$(dirname "$bag")" "$(basename "$bag")"
    echo "  archived: $(basename "$bag").tar.gz"
  fi
done

TARBALL="$OUT.tar.gz"
tar -czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "== done =="
ls -lh "$TARBALL"
echo "Copy this single file to Windows, into:"
echo "  C:\\Users\\Lenovo\\Desktop\\Personal\\RoutingComm\\LinuxTests\\"
