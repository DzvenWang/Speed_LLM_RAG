param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
python scripts/run_pipeline.py --stage development @Rest
