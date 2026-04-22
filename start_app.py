"""
Simple launcher for FairCheck India - Opens browser automatically
"""
import subprocess
import sys
import os
import webbrowser
import time
import threading

os.chdir(r'C:\Users\hp\Desktop\new hack')

print("=" * 50)
print("Starting FairCheck India...")
print("=" * 50)

def open_browser():
    time.sleep(5)
    url = "http://localhost:8530"
    try:
        webbrowser.open(url)
        print("Browser opened!")
    except:
        pass

browser_thread = threading.Thread(target=open_browser)
browser_thread.daemon = True
browser_thread.start()

proc = subprocess.Popen([
    sys.executable, '-m', 'streamlit', 'run', 'faircheck.py',
    '--server.port=8530',
    '--server.address=127.0.0.1',
    '--server.headless=false'
])

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    print("Stopped")