#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-outputs/d0_tiny_h3_fs5_delta_predaux_action_10k}"
RUN_NAME="${RUN_NAME:-d0_tiny_h3_fs5_delta_predaux_action_10k}"
TRAIN_PID="${TRAIN_PID:-}"
WATCH_INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-300}"
CARLA_ROOT="${CARLA_ROOT:-/home/ubuntu/carla/carla-ue4}"
CARLA_SESSION="${CARLA_SESSION:-carla2100}"
CARLA_PORT="${CARLA_PORT:-2100}"
CARLA_BOOT_SECONDS="${CARLA_BOOT_SECONDS:-60}"
PURE_EVAL_CONFIG="${PURE_EVAL_CONFIG:-configs/eval_d0_model_action_speedgate.yaml}"
LANE_EVAL_CONFIG="${LANE_EVAL_CONFIG:-configs/eval_d0_model_action_lane_keep_speedgate.yaml}"
PURE_EVAL_DIR="${PURE_EVAL_DIR:-outputs/d0_eval_tiny_model_action_speedgate_200m}"
LANE_EVAL_DIR="${LANE_EVAL_DIR:-outputs/d0_eval_tiny_model_action_lane_keep_speedgate_200m}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/watch_eval.log"
exec >>"$LOG_PATH" 2>&1

log() {
  echo "[$(date -Is)] $*"
}

pid_matches_run() {
  local pid="$1"
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" | grep -q "$RUN_NAME"
}

find_train_pid() {
  pgrep -af "carla_lewm_drive.driving_lewm.train" | grep "$RUN_NAME" | awk '{print $1; exit}'
}

emit_snapshot() {
  log "heartbeat"
  ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "${TRAIN_PID:-0}" 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw --format=csv,noheader 2>/dev/null || true
  tail -12 "$RUN_DIR/metrics.csv" 2>/dev/null || true
  find "$RUN_DIR" -maxdepth 1 -type f \( -name "*.pt" -o -name "test_metrics.json" \) -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort || true
  df -h . "$RUN_DIR" 2>/dev/null || true
}

if [[ -z "$TRAIN_PID" ]]; then
  TRAIN_PID="$(find_train_pid || true)"
fi

if [[ -z "$TRAIN_PID" ]]; then
  log "no matching train process found for $RUN_NAME"
else
  log "watching train pid $TRAIN_PID for $RUN_NAME"
  while pid_matches_run "$TRAIN_PID"; do
    emit_snapshot
    sleep "$WATCH_INTERVAL_SECONDS"
  done
  log "train pid $TRAIN_PID no longer matches $RUN_NAME"
fi

emit_snapshot

if [[ ! -f "$RUN_DIR/best.pt" ]]; then
  log "missing $RUN_DIR/best.pt; skip closed-loop evaluation"
  exit 2
fi

started_carla=0
if ! tmux has-session -t "$CARLA_SESSION" 2>/dev/null; then
  log "starting CARLA session $CARLA_SESSION on port $CARLA_PORT"
  tmux new-session -d -s "$CARLA_SESSION" "cd \"$CARLA_ROOT\" && ./CarlaUE4.sh -RenderOffScreen -nosound -world-port=$CARLA_PORT > /tmp/${CARLA_SESSION}.log 2>&1"
  started_carla=1
  sleep "$CARLA_BOOT_SECONDS"
else
  log "using existing CARLA session $CARLA_SESSION"
fi

export PYTHONPATH="${PYTHONPATH:-src}"
checkpoint_args=()
if [[ -n "$CHECKPOINT_PATH" ]]; then
  checkpoint_args=(--checkpoint "$CHECKPOINT_PATH")
fi

log "running pure model_action evaluation"
.venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config "$PURE_EVAL_CONFIG" \
  --output-dir "$PURE_EVAL_DIR" \
  "${checkpoint_args[@]}"

log "running model_action_lane_keep evaluation"
.venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config "$LANE_EVAL_CONFIG" \
  --output-dir "$LANE_EVAL_DIR" \
  "${checkpoint_args[@]}"

if [[ "$started_carla" == "1" ]]; then
  log "stopping CARLA session $CARLA_SESSION"
  tmux kill-session -t "$CARLA_SESSION" 2>/dev/null || true
fi

log "watch and evaluation complete"
