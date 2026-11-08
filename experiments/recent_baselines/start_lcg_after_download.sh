#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_PID=${1:?download PID required}
DOWNLOAD_LOG=/root/autodl-tmp/logs_recent_subset100/qwen_download.log
RUNNER=/root/autodl-tmp/legalfactaware_recent_baselines_run/experiments/recent_baselines_20261031/run_subset100_lcg.sh

while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
  sleep 30
done

if ! grep -q "Done\." "$DOWNLOAD_LOG"; then
  echo "Qwen download did not finish successfully; LCG run not started." >&2
  exit 1
fi

chmod +x "$RUNNER"
exec "$RUNNER"
