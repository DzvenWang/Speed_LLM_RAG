@echo off
set STAGE=development
if not "%~1"=="" set STAGE=%~1
set PYTHON=%USERPROFILE%\AppData\Local\miniconda3\envs\TFP\python.exe
if not exist "%PYTHON%" set PYTHON=python
"%PYTHON%" "%~dp0scripts\run_observation_experiments_local.py" --stage %STAGE%
exit /b %ERRORLEVEL%
