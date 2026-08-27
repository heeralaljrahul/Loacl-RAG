@echo off
REM Add reference documents to a campaign's memory.
REM   story-lore.bat reina "C:\path\to\character bible"
cd /d "%~dp0.."
call bat\_env.bat
.venv\Scripts\python play.py lore %1 %2 %3 %4 %5
pause
