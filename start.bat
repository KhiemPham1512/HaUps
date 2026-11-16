@echo off
setlocal
cd /d "%~dp0"
set "HAUPS_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%HAUPS_PYTHON%" set "HAUPS_PYTHON=%~dp0.local-artifacts\engine-poc\.venv\Scripts\python.exe"
if not exist "%HAUPS_PYTHON%" (
  echo Chua co moi truong Python. Xem README.md de cai dat.
  pause
  exit /b 1
)
rem Upload spool files (up to 1 GB per batch) and engine temp files stay on D, not the nearly full C drive.
if not exist "%~dp0.haups-data\tmp" mkdir "%~dp0.haups-data\tmp"
set "TEMP=%~dp0.haups-data\tmp"
set "TMP=%~dp0.haups-data\tmp"
echo Mo trinh duyet tai http://127.0.0.1:8765
echo Giu cua so nay mo de hang doi tiep tuc xu ly. Ctrl+C de dung ung dung.
"%HAUPS_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --workers 1
if errorlevel 1 pause
