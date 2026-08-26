@echo off
REM Usage:  ingest.bat "C:\Users\you\Documents\notes"
REM         ingest.bat            (rescans everything already added)
cd /d "%~dp0.."
call bat\_env.bat
.venv\Scripts\python cli.py ingest %*
pause
