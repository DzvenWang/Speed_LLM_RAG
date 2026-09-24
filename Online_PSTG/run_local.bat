@echo off
python scripts\preflight.py --stage development
if errorlevel 1 exit /b %errorlevel%
python scripts\run_online.py --stage development --variant head --max-queries 256
