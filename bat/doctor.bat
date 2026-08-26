@echo off
REM Checks Ollama, GPU placement, models and the index. Run this first when
REM something is wrong - it usually names the problem outright.
cd /d "%~dp0.."
call bat\_env.bat
.venv\Scripts\python cli.py doctor
pause
