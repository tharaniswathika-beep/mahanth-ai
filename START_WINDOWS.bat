@echo off
echo.
echo  ========================================
echo   Mahanth AI - Starting...
echo  ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found!
    echo  Download from: https://python.org
    pause
    exit
)

:: Install dependencies
echo  Installing required packages...
pip install -r requirements.txt -q

echo.
echo  Starting Mahanth AI...
echo  Open your browser at: http://localhost:5000
echo.
echo  Press Ctrl+C to stop
echo.

python app.py
pause
