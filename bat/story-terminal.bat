@echo off
REM Play in the terminal.   story-terminal.bat reina
cd /d "%~dp0.."
call bat\_env.bat
if "%~1"=="" (echo Usage: story-terminal.bat ^<campaign^> & pause & exit /b 1)
.venv\Scripts\python play.py play %1
pause
