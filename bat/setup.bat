@echo off
REM One-time setup. Run from the project folder by double-clicking.
cd /d "%~dp0.."

echo === Creating the virtual environment ===
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo Could not create a virtual environment. Is Python installed and on PATH?
  pause & exit /b 1
)

echo.
echo === Installing dependencies (about 300 MB, a few minutes) ===
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 ( echo Install failed. & pause & exit /b 1 )

echo.
echo === Making Ollama use less VRAM for context ===
setx OLLAMA_FLASH_ATTENTION 1 >nul
setx OLLAMA_KV_CACHE_TYPE q8_0 >nul
setx OLLAMA_KEEP_ALIVE 30m >nul
echo Set. IMPORTANT: quit Ollama from the tray and start it again -
echo these only apply to a freshly started server.

echo.
echo === Downloading the embedding and rerank models (about 130 MB) ===
.venv\Scripts\python cli.py doctor

echo.
echo Done. Next: ingest.bat "C:\path\to\your\documents"
pause
