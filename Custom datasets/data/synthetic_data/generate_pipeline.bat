@echo off
REM Batch wrapper that runs the PowerShell pipeline under the local venv
setlocal enabledelayedexpansion

REM Resolve this script directory (handles spaces in path)
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"

REM Create venv if missing
if not exist ".\venv\Scripts\activate.bat" (
    echo Creating Python virtual environment...
    python -m venv venv
)

call ".\venv\Scripts\activate.bat"

REM Ensure required packages are installed
python -m pip install --upgrade pip >nul 2>&1
python -m pip install pandas tqdm requests kaggle >nul 2>&1

REM Run the pipeline PowerShell script located in the same directory
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_pipeline.ps1"

popd
pause

