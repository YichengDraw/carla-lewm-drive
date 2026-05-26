param(
  [ValidateSet("probe", "sync", "start-carla", "eval-ft35-filter", "train-ft37", "monitor-ft37", "all")]
  [string]$Step = "probe",
  [string]$HostName = "ubuntu@10.26.4.44",
  [string]$RemoteRoot = "/home/ubuntu/carla_lewm_drive",
  [string]$CarlaRoot = "/home/ubuntu/carla/carla-ue4",
  [int]$CarlaPort = 2100,
  [string]$CarlaSession = "",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CarlaSession)) {
  $CarlaSession = "carla_$CarlaPort"
}

$sshOptions = @(
  "-F", "NUL",
  "-o", "BatchMode=yes",
  "-o", "ConnectTimeout=10",
  "-o", "ConnectionAttempts=1",
  "-o", "ServerAliveInterval=15",
  "-o", "ServerAliveCountMax=2"
)

function Invoke-Remote {
  param([Parameter(Mandatory=$true)][string]$Command)
  $argv = @($sshOptions + @($HostName, $Command))
  Write-Host "ssh.exe $($argv -join ' ')"
  if (-not $DryRun) {
    & ssh.exe @argv
    if ($LASTEXITCODE -ne 0) {
      throw "ssh command failed with exit code $LASTEXITCODE"
    }
  }
}

function Copy-ToRemote {
  param(
    [Parameter(Mandatory=$true)][string]$LocalPath,
    [Parameter(Mandatory=$true)][string]$RemotePath
  )
  $remoteDir = Split-Path -Path $RemotePath -Parent
  $remoteDir = $remoteDir -replace "\\", "/"
  Invoke-Remote "mkdir -p '$remoteDir'"
  $argv = @("-F", "NUL", "-o", "ConnectTimeout=10", $LocalPath, "${HostName}:$RemotePath")
  Write-Host "scp.exe $($argv -join ' ')"
  if (-not $DryRun) {
    & scp.exe @argv
    if ($LASTEXITCODE -ne 0) {
      throw "scp failed for $LocalPath with exit code $LASTEXITCODE"
    }
  }
}

function Step-Probe {
  $cmd = @'
cd __REMOTE_ROOT__ && \
hostname && date -Is && \
git rev-parse --short HEAD 2>/dev/null || true; \
test -x /home/ubuntu/carla/carla-ue4/CarlaUE4.sh && echo CARLA_BINARY_OK || echo CARLA_BINARY_MISSING; \
test -f configs/eval_d1_route6_perception_lane_keep_ft37_robustfilter_slow_lg080_hg060_tick_1km.yaml && echo FT37_EVAL_CONFIG_OK || echo FT37_EVAL_CONFIG_MISSING; \
test -f configs/train_d1_tiny_route6_perception_lane_ft37_fs1_temporal_aux_tail220_bce_sign_2400.yaml && echo FT37_TRAIN_CONFIG_OK || echo FT37_TRAIN_CONFIG_MISSING; \
tmux ls 2>/dev/null || true; \
ss -ltnp 2>/dev/null | grep -E ":2100|:2110|:__CARLA_PORT__" || true; \
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true; \
df -h /home __REMOTE_ROOT__ 2>/dev/null || true
'@
  $cmd = $cmd.Replace("__REMOTE_ROOT__", $RemoteRoot).Replace("__CARLA_PORT__", [string]$CarlaPort)
  Invoke-Remote $cmd
}

function Step-Sync {
  $files = @(
    "src/carla_lewm_drive/driving_lewm/model.py",
    "src/carla_lewm_drive/closed_loop_eval/evaluate.py",
    "configs/eval_d1_route6_perception_lane_keep_ft37_robustfilter_slow_lg080_hg060_tick_1km.yaml",
    "configs/train_d1_tiny_route6_perception_lane_ft37_fs1_temporal_aux_tail220_bce_sign_2400.yaml"
  )
  foreach ($file in $files) {
    $remote = "$RemoteRoot/$($file -replace '\\', '/')"
    Copy-ToRemote -LocalPath $file -RemotePath $remote
  }
}

function Step-StartCarla {
  $cmd = @'
cd __REMOTE_ROOT__ && mkdir -p outputs/remote_logs && \
if ss -ltn 2>/dev/null | grep -q ":__CARLA_PORT__"; then \
  echo CARLA_PORT_ALREADY_LISTENING; ss -ltnp 2>/dev/null | grep ":__CARLA_PORT__" || true; exit 0; \
fi; \
LOG=outputs/remote_logs/carla___CARLA_PORT___$(date +%Y%m%d_%H%M%S).log; \
tmux new-session -d -s __CARLA_SESSION__ "cd __CARLA_ROOT__ && ./CarlaUE4.sh -RenderOffScreen -nosound -world-port=__CARLA_PORT__ > __REMOTE_ROOT__/$LOG 2>&1"; \
for i in $(seq 1 60); do \
  if ss -ltn 2>/dev/null | grep -q ":__CARLA_PORT__"; then echo CARLA_READY:$i; break; fi; \
  sleep 2; \
done; \
ss -ltnp 2>/dev/null | grep ":__CARLA_PORT__" || true; \
tail -n 60 "$LOG" 2>/dev/null || true
'@
  $cmd = $cmd.Replace("__REMOTE_ROOT__", $RemoteRoot).
    Replace("__CARLA_ROOT__", $CarlaRoot).
    Replace("__CARLA_PORT__", [string]$CarlaPort).
    Replace("__CARLA_SESSION__", $CarlaSession)
  Invoke-Remote $cmd
}

function Step-EvalFt35Filter {
  $cmd = @'
cd __REMOTE_ROOT__ && bash -lc 'set -o pipefail; mkdir -p outputs/remote_logs; TS=$(date +%Y%m%d_%H%M%S); OUT=outputs/d1_eval_ft35_retry_best_robustfilter_route6_1km_${TS}; LOG=outputs/remote_logs/eval_ft35_retry_best_robustfilter_route6_1km_${TS}.log; env PYTHONPATH=src PYTHONFAULTHANDLER=1 .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate --config configs/eval_d1_route6_perception_lane_keep_ft37_robustfilter_slow_lg080_hg060_tick_1km.yaml --checkpoint outputs/d1_tiny_h1_fs1_route6_perception_lane_ft35_temporal_aux_tail220_sign_1800_retry/best.pt --output-dir ${OUT} 2>&1 | tee ${LOG}; rc=${PIPESTATUS[0]}; echo EXIT_CODE:${rc} | tee -a ${LOG}; cat ${OUT}/summary.json 2>/dev/null || true; cat ${OUT}/episodes.csv 2>/dev/null || true; exit ${rc}'
'@
  $cmd = $cmd.Replace("__REMOTE_ROOT__", $RemoteRoot)
  Invoke-Remote $cmd
}

function Step-TrainFt37 {
  $cmd = @'
cd __REMOTE_ROOT__ && bash -lc 'set -euo pipefail; mkdir -p outputs/remote_logs; test -f "$HOME/.netrc"; .venv/bin/python -c "import wandb; print(\"wandb_import_ok\")"; if tmux has-session -t train_ft37_bce_sign 2>/dev/null; then echo TRAIN_SESSION_ALREADY_EXISTS; tmux capture-pane -pt train_ft37_bce_sign -S -80 || true; exit 2; fi; LOG=outputs/remote_logs/train_ft37_temporal_aux_tail220_bce_sign_$(date +%Y%m%d_%H%M%S).log; tmux new-session -d -s train_ft37_bce_sign "cd __REMOTE_ROOT__ && env PYTHONPATH=src PYTHONFAULTHANDLER=1 .venv/bin/python -m carla_lewm_drive.driving_lewm.train --config configs/train_d1_tiny_route6_perception_lane_ft37_fs1_temporal_aux_tail220_bce_sign_2400.yaml 2>&1 | tee $LOG"; echo TRAIN_LOG:$LOG; sleep 8; tmux ls 2>/dev/null || true; tail -n 100 $LOG'
'@
  $cmd = $cmd.Replace("__REMOTE_ROOT__", $RemoteRoot)
  Invoke-Remote $cmd
}

function Step-MonitorFt37 {
  $cmd = @'
cd __REMOTE_ROOT__ && \
date -Is; \
tmux ls 2>/dev/null || true; \
ps -eo pid,ppid,stat,etime,cmd | grep -E "driving_lewm.train|train_ft37|wandb" | grep -v grep || true; \
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw --format=csv,noheader 2>/dev/null || true; \
find outputs/d1_tiny_h1_fs1_route6_perception_lane_ft37_temporal_aux_tail220_bce_sign_2400 -maxdepth 1 -type f \( -name "*.pt" -o -name "metrics.csv" -o -name "test_metrics.json" \) -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort || true; \
tail -n 40 outputs/d1_tiny_h1_fs1_route6_perception_lane_ft37_temporal_aux_tail220_bce_sign_2400/metrics.csv 2>/dev/null || true; \
ls -t outputs/remote_logs/train_ft37_temporal_aux_tail220_bce_sign_*.log 2>/dev/null | head -1 | xargs -r tail -n 120
'@
  $cmd = $cmd.Replace("__REMOTE_ROOT__", $RemoteRoot)
  Invoke-Remote $cmd
}

switch ($Step) {
  "probe" { Step-Probe }
  "sync" { Step-Sync }
  "start-carla" { Step-StartCarla }
  "eval-ft35-filter" { Step-EvalFt35Filter }
  "train-ft37" { Step-TrainFt37 }
  "monitor-ft37" { Step-MonitorFt37 }
  "all" {
    Step-Probe
    Step-Sync
    Step-StartCarla
    Step-EvalFt35Filter
  }
}
