@echo off
cd /d "%~dp0"
echo ================================================
echo   FairCheck India - Starting...
echo ================================================
echo.
echo Opening browser in 3 seconds...
timeout /nobreak /t 3 >nul
start http://127.0.0.1:8510
start http://localhost:8510
echo.
echo If browser doesn't open, go to:
echo   http://127.0.0.1:8510
echo   http://localhost:8510
echo.
echo ================================================
streamlit run faircheck.py --server.port=8510
pause