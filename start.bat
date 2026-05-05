@echo off
title FinAnalyst NLP Platform

echo.
echo ==========================================
echo   FinAnalyst NLP Platform v1.0
echo   Finance Q^&A + Document Analyzer
echo ==========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python is not installed. Get it from https://python.org
  pause
  exit /b 1
)

echo Installing backend dependencies...
cd /d "%~dp0backend"
pip install -r requirements.txt -q

echo.
echo Starting FastAPI backend on http://localhost:8000 ...
start /B python main.py

timeout /t 3 /nobreak >nul

echo Starting frontend on http://localhost:3000 ...
cd /d "%~dp0frontend"
start /B python -m http.server 3000

echo.
echo ==========================================
echo   FinAnalyst is running!
echo.
echo   Open browser:  http://localhost:3000
echo   API docs:      http://localhost:8000/docs
echo ==========================================
echo.
echo Press any key to stop...
pause >nul

taskkill /F /IM python.exe /T >nul 2>&1
