param(
  [string]$HostName = "ubuntu@10.26.4.44",
  [string]$RemoteRoot = "/home/ubuntu/carla_lewm_drive",
  [int]$PollSeconds = 60,
  [int]$MonitorSeconds = 600,
  [double]$MaxHours = 12.0,
  [switch]$NoTrain
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Recovery = Join-Path $ScriptDir "remote_route6_ft37_recovery.ps1"
$LogDir = Join-Path (Split-Path -Parent $ScriptDir) "outputs\local_watch"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "route6_recovery_watch_$Stamp.log"

$sshOptions = @(
  "-F", "NUL",
  "-o", "BatchMode=yes",
  "-o", "ConnectTimeout=8",
  "-o", "ConnectionAttempts=1",
  "-o", "ServerAliveInterval=15",
  "-o", "ServerAliveCountMax=2"
)

function Write-WatchLog {
  param([Parameter(Mandatory=$true)][string]$Message)
  $line = "$(Get-Date -Format o) $Message"
  Add-Content -Path $LogPath -Value $line
  Write-Host $line
}

function Test-RemoteSsh {
  $argv = @($sshOptions + @($HostName, "date -Is"))
  $oldPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $out = & ssh.exe @argv 2>&1
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $oldPreference
  }
  $ok = $code -eq 0
  if ($ok) {
    Write-WatchLog "ssh_ok $($out -join ' ')"
  } else {
    Write-WatchLog "ssh_not_ready exit=$code $($out -join ' ')"
  }
  return $ok
}

function Invoke-RecoveryStep {
  param([Parameter(Mandatory=$true)][string]$Step)
  Write-WatchLog "step_begin $Step"
  $oldPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Recovery -Step $Step -HostName $HostName 2>&1 |
      Tee-Object -FilePath $LogPath -Append
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $oldPreference
  }
  if ($code -ne 0) {
    throw "recovery step failed: $Step exit=$code"
  }
  Write-WatchLog "step_ok $Step"
}

function Get-LatestEvalSummary {
  $cmd = @'
cd __REMOTE_ROOT__ && python3 - <<'PY'
import glob, json, os
paths = sorted(
    glob.glob("outputs/d1_eval_ft35_retry_best_robustfilter_route6_1km_*/summary.json"),
    key=os.path.getmtime,
)
if not paths:
    print(json.dumps({"ok": False, "reason": "no_summary"}))
    raise SystemExit(0)
path = paths[-1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
data["summary_path"] = path
data["ok"] = True
print(json.dumps(data, sort_keys=True))
PY
'@
  $cmd = $cmd.Replace("__REMOTE_ROOT__", $RemoteRoot)
  $argv = @($sshOptions + @($HostName, $cmd))
  $oldPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    $out = & ssh.exe @argv 2>&1
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $oldPreference
  }
  if ($code -ne 0) {
    throw "summary read failed exit=$code $($out -join ' ')"
  }
  $text = ($out -join "`n").Trim()
  Write-WatchLog "latest_eval_summary $text"
  return $text | ConvertFrom-Json
}

function Test-OneKmPass {
  param([Parameter(Mandatory=$true)]$Summary)
  if (-not $Summary.ok) {
    return $false
  }
  $ifd = [double]$Summary.mean_infraction_free_distance_m
  $primary = [double]$Summary.success_rate_no_primary_safety_infraction
  $red = [double]$Summary.success_rate_no_primary_or_red_infraction
  return ($ifd -ge 999.0 -and $primary -ge 1.0 -and $red -ge 1.0)
}

$deadline = (Get-Date).AddHours($MaxHours)
Write-WatchLog "watch_start host=$HostName max_hours=$MaxHours poll_s=$PollSeconds monitor_s=$MonitorSeconds no_train=$NoTrain"

while ((Get-Date) -lt $deadline) {
  if (Test-RemoteSsh) {
    try {
      Invoke-RecoveryStep -Step "probe"
      Invoke-RecoveryStep -Step "sync"
      Invoke-RecoveryStep -Step "start-carla"
      Invoke-RecoveryStep -Step "eval-ft35-filter"
      $summary = Get-LatestEvalSummary
      if (Test-OneKmPass -Summary $summary) {
        Write-WatchLog "goal_candidate_pass_ft35_filter summary_path=$($summary.summary_path)"
        exit 0
      }
      Write-WatchLog "ft35_filter_not_pass ifd=$($summary.mean_infraction_free_distance_m) primary=$($summary.success_rate_no_primary_safety_infraction)"
      if ($NoTrain) {
        Write-WatchLog "no_train_set_stop_after_eval"
        exit 2
      }
      Invoke-RecoveryStep -Step "train-ft37"
      while ((Get-Date) -lt $deadline) {
        Invoke-RecoveryStep -Step "monitor-ft37"
        Start-Sleep -Seconds $MonitorSeconds
      }
      exit 0
    } catch {
      Write-WatchLog "recovery_error $($_.Exception.Message)"
    }
  }
  Start-Sleep -Seconds $PollSeconds
}

Write-WatchLog "watch_timeout"
exit 124
