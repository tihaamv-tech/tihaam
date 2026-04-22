@echo off
cd /d "%~dp0"
start http://localhost:8530
python -m streamlit run faircheck.py --server.port=8530
pause