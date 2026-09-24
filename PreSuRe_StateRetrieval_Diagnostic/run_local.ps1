param(
  [ValidateSet("development","formal")][string]$Stage="development",
  [int]$Queries=256,
  [int]$MaxWindows=12000,
  [int]$TopK=50,
  [string]$Scopes="1,3,6,12",
  [string]$OracleKValues="1,5,10,20,50",
  [string]$OracleTransport="affine_alignment",
  [string]$TransportMethods="raw_copy,delta_anchor,affine_alignment",
  [string]$WeightedK="5,10",
  [int]$TransportOracleK=20,
  [string]$ForecastOracleTransport="affine_alignment",
  [string]$ForecastOracleKValues="1,5,10,20,50",
  [int]$SameEpisodePoolCap=512,
  [int]$EvalBatchSize=64,
  [switch]$PreflightOnly
)
$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $MyInvocation.MyCommand.Path
$Python=$env:PRESURE_SRD_PYTHON
if (-not $Python) {
  $Candidate=Join-Path $env:USERPROFILE "AppData\Local\miniconda3\envs\TFP\python.exe"
  if (Test-Path $Candidate) { $Python=$Candidate } else { $Python="python" }
}
Push-Location $Root
try {
  $ArgsList=@(
    "scripts/run_local_pipeline.py","--stage",$Stage,"--queries",$Queries,
    "--max-windows",$MaxWindows,"--top-k",$TopK,"--scopes",$Scopes,
    "--oracle-k-values",$OracleKValues,"--oracle-transport",$OracleTransport,
    "--transport-methods",$TransportMethods,"--weighted-k",$WeightedK,
    "--transport-oracle-k",$TransportOracleK,
    "--forecast-oracle-transport",$ForecastOracleTransport,
    "--forecast-oracle-k-values",$ForecastOracleKValues,
    "--same-episode-pool-cap",$SameEpisodePoolCap,
    "--eval-batch-size",$EvalBatchSize
  )
  if ($PreflightOnly) { $ArgsList += @("--preflight-only") }
  & $Python @ArgsList
  exit $LASTEXITCODE
} finally { Pop-Location }
