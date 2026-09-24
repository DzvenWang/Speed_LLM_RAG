@echo off
setlocal
set "PYTHON_EXE=%PRESURE_SRD_PYTHON%"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
"%PYTHON_EXE%" "%~dp0scripts\run_local_pipeline.py" %*
exit /b %ERRORLEVEL%
