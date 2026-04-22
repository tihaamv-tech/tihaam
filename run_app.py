"""
Simple launcher for FairCheck India
"""
import subprocess
import sys
import os
import webbrowser
import time

# Change to app directory
os.chdir(r'C:\Users\hp\Desktop\new hack')

print("=" * 60)
print("   FairCheck India - Starting...")
print("=" * 60)

# Kill any existing streamlit processes
try:
    subprocess.run(['taskkill', '/F', '/IM', 'streamlit.exe'], 
               capture_output=True, timeout=5)
except:
    pass

time.sleep(2)

# Start Streamlit
print("\nStarting server on http://localhost:8511")
proc = subprocess.Popen([
    sys.executable, '-m', 'streamlit', 'run', 'faircheck.py',
    '--server.port=8511',
    '--server.address=127.0.0.1',
    '--server.headless=false'
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for server to start
time.sleep(8)

# Try to open browser
print("Opening browser...")
webbrowser.open('http://localhost:8511')
webbrowser.open('http://127.0.0.1:8511')

print("\n   If browser doesn't open, manually go to:")
print("   http://localhost:8511")
print("   http://127.0.0.1:8511")
print("=" * 60)
print("\nPress Ctrl+C to stop server")

# Keep running
try:
    proc.wait()
except KeyboardInterrupt:
    print("\nStopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()
    print("Server stopped")