param(
  [ValidateSet('development','formal')][string]$Stage='development',
  [string]$RunTag='',
  [switch]$ForcePrepare
)
$ArgsList = @('scripts\run_p1.py','--stage',$Stage)
if ($RunTag -ne '') { $ArgsList += @('--run-tag',$RunTag) }
if ($ForcePrepare) { $ArgsList += '--force-prepare' }
python @ArgsList
