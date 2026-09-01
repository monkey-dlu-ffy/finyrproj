param(
  [string]$Model = "qwen2.5:3b-instruct",
  [int]$MaxAttempts = 400,
  [int]$MaxHours = 12,
  [int]$StallMinutes = 6,
  [int]$PollSeconds = 60
)

# Supervised `ollama pull` for a link that cannot hold a long transfer.
#
# Both Phase 0 models were pulled with this. A plain `ollama pull` never
# finished; a naive retry loop finished 3b but hung on 7b, because there are
# two distinct failure modes and the naive loop only handled the first:
#
#   1. `ollama pull` exits non-zero (TLS handshake timeout)   -> retry
#   2. `ollama pull` HANGS, moving no bytes but never exiting -> the retry loop
#      blocks forever. This supervisor watches actual byte progress and kills a
#      stalled child so the retry can restart it.
#
# Retrying is cheap: Ollama resumes from the partial blob, so each attempt makes
# forward progress rather than restarting from zero.
#
# Progress is measured by free-disk delta, NOT by the partial file's length,
# because Ollama pre-allocates the blob to its full size up front -- the length
# is constant from the first second and says nothing about bytes fetched. The
# 1 MB threshold below absorbs unrelated disk churn from other processes.
#
# Do NOT redirect the native command's stderr: in PowerShell 5.1 that wraps each
# line in a NativeCommandError and corrupts exit-code handling.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\pull_model.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\pull_model.ps1 -Model qwen2.5:7b-instruct
#
# Exit codes: 0 pulled | 1 attempts exhausted | 2 deadline | 3 disk too low | 4 ollama not found

$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
  $onPath = Get-Command ollama -ErrorAction SilentlyContinue
  if ($onPath) { $ollama = $onPath.Source }
  else {
    Write-Output "=== ABORT: ollama.exe not found (winget install Ollama.Ollama) ==="
    exit 4
  }
}

$deadline = (Get-Date).AddHours($MaxHours)

function Get-FreeMB { [math]::Round((Get-PSDrive C).Free/1MB, 1) }
function Stamp { (Get-Date).ToString('HH:mm:ss') }

Write-Output "=== supervised pull: $Model ==="
Write-Output "ollama       : $ollama"
Write-Output "started      : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "deadline     : $($deadline.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output "stall kill   : no disk progress for $StallMinutes min"
Write-Output "free disk    : $([math]::Round((Get-FreeMB)/1024,2)) GB"
Write-Output ""

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {

  if ((Get-Date) -gt $deadline) {
    Write-Output "=== DEADLINE reached after $($attempt-1) attempts ==="
    exit 2
  }

  # Stop while the disk is still usable rather than filling it completely.
  $freeGB = (Get-FreeMB)/1024
  if ($freeGB -lt 2.0) {
    Write-Output "=== ABORT: only $([math]::Round($freeGB,2)) GB free ==="
    exit 3
  }

  Write-Output "[$(Stamp)] attempt $attempt starting (free $([math]::Round($freeGB,2)) GB)"

  $proc = Start-Process -FilePath $ollama -ArgumentList "pull",$Model `
            -WindowStyle Hidden -PassThru

  $lastFree     = Get-FreeMB
  $lastProgress = Get-Date
  $downloaded   = 0.0

  while ($true) {
    Start-Sleep -Seconds $PollSeconds

    if ($proc.HasExited) { break }

    $now   = Get-FreeMB
    $delta = $lastFree - $now          # positive == disk consumed == progress

    if ($delta -gt 1.0) {
      $downloaded += $delta
      $rate = [math]::Round($delta/$PollSeconds, 2)
      Write-Output ("[{0}]   +{1} MB ({2} MB/s), {3} MB this attempt" -f `
        (Stamp), [math]::Round($delta,0), $rate, [math]::Round($downloaded,0))
      $lastProgress = Get-Date
      $lastFree = $now
    }
    else {
      $stalledFor = ((Get-Date) - $lastProgress).TotalMinutes
      if ($stalledFor -ge $StallMinutes) {
        Write-Output ("[{0}]   STALLED {1} min with no bytes -- killing pid {2}" -f `
          (Stamp), [math]::Round($stalledFor,1), $proc.Id)
        try { Stop-Process -Id $proc.Id -Force -ErrorAction Stop } catch {}
        Start-Sleep -Seconds 5
        break
      }
    }
  }

  # Give a killed process a moment to release the blob handle.
  if (-not $proc.HasExited) { Start-Sleep -Seconds 5 }
  $code = if ($proc.HasExited) { $proc.ExitCode } else { -1 }

  if ($code -eq 0) {
    Write-Output "[$(Stamp)] attempt $attempt SUCCEEDED"
    Write-Output "=== COMPLETE at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    & $ollama list | Out-String | Write-Output
    exit 0
  }

  Write-Output "[$(Stamp)] attempt $attempt ended (exit $code), retrying"
  Start-Sleep -Seconds 15
}

Write-Output "=== GAVE UP after $MaxAttempts attempts ==="
exit 1
