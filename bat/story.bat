@echo off
REM Play in the browser.   story.bat reina
cd /d "%~dp0.."
call bat\_env.bat
if "%~1"=="" (echo Usage: story.bat ^<campaign^> & pause & exit /b 1)
start "" http://localhost:8090
.venv\Scripts\python play.py serve %1 --port 8090
pause
