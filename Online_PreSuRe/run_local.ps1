param([string]$Variant="rag_utility_compositional", [int]$MaxQueries=0)
$cmd = @("scripts/run_pipeline.py", "--stage", "development", "--variant", $Variant)
if ($MaxQueries -gt 0) { $cmd += @("--max-queries", "$MaxQueries") }
python @cmd
