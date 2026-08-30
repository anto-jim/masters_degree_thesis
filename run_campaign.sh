#!/usr/bin/env bash
# Detached launcher for the canonical multi-seed campaign.
# Everything runs at num_turns=5 for every algorithm (see thesis 4.1.2).
set -uo pipefail

# Resolve the repository root from this script's own location, so a fresh
# checkout reproduces the campaign without editing any path in here.
REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
cd "$REPO_ROOT"

# Override PYTHON to use an interpreter outside the campaign venv.
PYTHON="${PYTHON:-$REPO_ROOT/venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "no usable interpreter at $PYTHON." >&2
  echo "Create the campaign venv, or set PYTHON=/path/to/python." >&2
  exit 1
fi

ROOT="${1:-turn_battle_results}"
LOG="$ROOT/campaign.log"

# Refuse to start a second campaign against the same output root: two runs
# writing the same seed directories silently corrupt checkpoints, learner logs,
# and training curves.
existing=$(pgrep -f "turn_battle_marl_study.py .*--results_root=$ROOT" | grep -v "^$$\$" || true)
if [ -n "$existing" ]; then
  echo "refusing to start: a campaign is already running for $ROOT (pids: $existing)" >&2
  exit 1
fi

# Refuse to overwrite a completed campaign unless explicitly forced. The
# canonical root also holds VERIFICATION_REPORT.md, which is authored by hand and
# must never be destroyed by a rerun, so only generated subtrees are cleared.
if compgen -G "$ROOT/seed_*" >/dev/null; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "refusing to start: $ROOT already holds campaign results." >&2
    echo "Re-run with FORCE=1 to clear them, or pass a different root:" >&2
    echo "  FORCE=1 $0 $ROOT        # overwrite in place" >&2
    echo "  $0 turn_battle_results_rerun   # write elsewhere" >&2
    exit 1
  fi
  echo "FORCE=1: clearing previous seed results under $ROOT" >&2
  rm -rf "$ROOT"/seed_* "$ROOT"/figures "$ROOT"/aggregated_results.json \
         "$ROOT"/thesis_aggregated_report.tex "$LOG"
fi

mkdir -p "$ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/build/python"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$REPO_ROOT/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"

echo "=== campaign start $(date -Is) pid=$$ root=$REPO_ROOT ===" >>"$LOG"
"$PYTHON" -u open_spiel/python/examples/turn_battle_marl_study.py \
  --mode=multi_seed \
  --results_root="$ROOT" \
  --seeds=42,43,44 \
  --train_episodes=300 \
  --eval_episodes=50 \
  --eval_every=50 \
  --bracket_size=4 \
  --game_params=num_turns=5 \
  --dcfr_max_turns=0 \
  --round_robin=true \
  --device=auto >>"$LOG" 2>&1
status=$?
echo "=== campaign exit=$status $(date -Is) ===" >>"$LOG"
