@echo off
setlocal
rem Use the active environment, or an explicitly configured interpreter.
if not defined CALIB_PYTHON set "CALIB_PYTHON=python"
"%CALIB_PYTHON%" "%~dp0calibrate.py" %*
set "CALIB_EXIT=%ERRORLEVEL%"
endlocal & exit /b %CALIB_EXIT%
