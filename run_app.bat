@echo off
REM Quick Start Script for Streamlit App (Windows)
REM Usage: Double-click this file to start the Streamlit app

echo.
echo ========================================
echo  Climate Change and Food Security
echo  Nigeria Analysis Dashboard
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/Update dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

cls
echo.
echo ========================================
echo  Starting Streamlit Application
echo ========================================
echo.
echo Starting server at http://localhost:8501
echo.
echo To stop the app, press Ctrl+C
echo.

streamlit run app.py
