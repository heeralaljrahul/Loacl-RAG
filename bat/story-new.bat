@echo off
REM Create a campaign.   story-new.bat reina
REM Optionally seed from your own opening prose:
REM   story-new.bat reina "C:\path\to\opening.txt"
cd /d "%~dp0.."
call bat\_env.bat
if "%~2"=="" (
  .venv\Scripts\python play.py new %1 --seed seeds\reina.json
) else (
  .venv\Scripts\python play.py new %1 --seed seeds\reina.json --opening "%~2"
)
pause
