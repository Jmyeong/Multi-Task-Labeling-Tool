@echo off
setlocal
if not defined PYTHON set "PYTHON=python"
set "PYTHONUTF8=1"
"%PYTHON%" "%~dp0run.py" pseudo %*
exit /b %errorlevel%
