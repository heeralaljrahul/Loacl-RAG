@echo off
REM Web UI at http://localhost:8080
cd /d "%~dp0.."
call bat\_env.bat
start "" http://localhost:8080
.venv\Scripts\python cli.py serve --port 8080
pause
