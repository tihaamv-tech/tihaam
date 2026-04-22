@echo off
title FairCheck India
cd /d "%~dp0"
echo ================================================
echo   Starting FairCheck India...
echo ================================================
start /b python -m streamlit run faircheck.py --server.port=8530 --server.headless=false
timeout /nobreak /t 8 >nul
start http://localhost:8530
echo.
echo If browser did not open automatically, go to:
echo http://localhost:8530
echo.
pause