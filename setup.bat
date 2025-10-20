@echo off
REM Windows Setup Script for Devign Repository
REM Run this after cloning the repository

echo ========================================
echo DEVIGN REPOSITORY SETUP - WINDOWS
echo ========================================

REM Check if we're in the right directory
if not exist "main.py" (
    echo ERROR: main.py not found!
    echo Please run this script from the Devign repository root directory
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Step 2: Activating virtual environment...
call venv\Scripts\activate

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo Step 4: Installing dependencies...
pip install -r requirements-hyperopt.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo Step 5: Creating data directories...
mkdir data\raw 2>nul
mkdir data\cpg 2>nul
mkdir data\tokens 2>nul
mkdir data\input 2>nul
mkdir data\w2v 2>nul
mkdir data\model 2>nul

echo Step 6: Testing environment...
python test_environment.py
if errorlevel 1 (
    echo WARNING: Environment test failed
)

echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Download and extract Joern to joern\joern-cli\
echo 2. Update configs.json with correct Joern path
echo 3. Place your dataset in data\raw\
echo 4. Run: python main.py --create --embed --process
echo.
echo To activate environment in future sessions:
echo venv\Scripts\activate
echo.
pause