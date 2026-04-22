@echo off
cd /d "%~dp0"
python -m streamlit run faircheck.py --server.port=8530
pause