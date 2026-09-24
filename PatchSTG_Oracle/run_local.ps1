param(
  [ValidateSet('development','formal')][string]$Stage='development',
  [string]$RunTag='',
  [string]$Checkpoint='',
  [switch]$SkipPreflight,
  [switch]$PreflightOnly
)
$TfpPython=Join-Path $env:USERPROFILE 'AppData\Local\miniconda3\envs\TFP\python.exe'
$Python=if (Test-Path $TfpPython) { $TfpPython } else { 'python' }
$Runner=Join-Path $PSScriptRoot 'scripts\run_observation_experiments_local.py'
$ArgsList=@($Runner,'--stage',$Stage)
if ($RunTag) { $ArgsList += @('--run-tag',$RunTag) }
if ($Checkpoint) { $ArgsList += @('--checkpoint',$Checkpoint) }
if ($SkipPreflight) { $ArgsList += '--skip-preflight' }
if ($PreflightOnly) { $ArgsList += '--preflight-only' }
& $Python @ArgsList
exit $LASTEXITCODE
