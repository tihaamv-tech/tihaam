"""Simple FairCheck India Launcher"""
import os
import webbrowser
import time
import subprocess

# Change to app directory  
os.chdir(r"C:\Users\hp\Desktop\new hack")

print("=" * 50)
print("FairCheck India - Launcher")
print("=" * 50)

# Try to kill old processes
try:
    subprocess.run(["taskkill", "/F", "/IM", "streamlit.exe"], 
               capture_output=True, timeout=3)
except:
    pass

print("Starting server...")
time.sleep(2)

# Start fresh server on port 8515
proc = subprocess.Popen([
    "streamlit", "run", "faircheck.py",
    "--server.port=8515",
    "--server.headless=false"
])

# Wait for startup
time.sleep(10)

# Try multiple browser openings
urls = [
    "http://127.0.0.1:8515",
    "http://localhost:8515", 
    "http://127.0.0.1:8515/?core=true",
    "http://localhost:8515/?core=true"
]

for url in urls:
    try:
        webbrowser.open(url)
        print(f"Opened: {url}")
    except:
        pass

time.sleep(2)

print("=" * 50)
print("Server should be open in your browser!")
print("If not, manually open: http://127.0.0.1:8515")
print("=" * 50)

# Keep process running
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    print("Server stopped")