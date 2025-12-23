@echo off
REM AI Data Platform 2025 - Windows Startup Script
REM Double-click this file to start the application

echo ============================================================
echo    AI Data Platform 2025 - Starting...
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Start the application
echo Starting the application...
echo.
python modern_ui_complete.py

REM If the script exits, pause to show any error messages
if errorlevel 1 (
    echo.
    echo ============================================================
    echo    Application stopped with an error
    echo ============================================================
    pause
)
