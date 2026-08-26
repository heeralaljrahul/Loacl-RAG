@echo off
REM Interactive terminal chat over your documents.
cd /d "%~dp0.."
call bat\_env.bat
.venv\Scripts\python cli.py chat
pause
