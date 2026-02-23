@echo off
TITLE Stock Analyzer AI Backend
COLOR 0A

echo ========================================================
echo       STOCK ANALYZER AI - ONE CLICK LAUNCHER
echo ========================================================
echo.
echo Starting Backend Server...
echo.

REM Open browser after 3 seconds (gives server time to start)
start "" cmd /c "timeout /t 3 >nul && start http://127.0.0.1:5001/login.html"

cd backend
python server.py

echo.
echo ========================================================
echo Server has stopped or crashed.
echo Check the error message above.
echo ========================================================
pause
