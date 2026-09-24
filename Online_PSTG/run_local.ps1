python scripts/preflight.py --stage development
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_online.py --stage development --variant head --max-queries 256
