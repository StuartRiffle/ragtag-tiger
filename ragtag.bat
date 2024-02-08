@echo off
setlocal, enabledelayedexpansion

if not exist "%~dp0.venv" (
    echo Creating virtual environment...
    python -m venv "%~dp0.venv" || exit /b 1
    call "%~dp0.venv\Scripts\activate.bat"

    echo Installing requirements...
    pip install --upgrade pip
    pip install -r "%~dp0requirements.txt" || exit /b 1

    echo Testing installation...
    python "%~dp0ragtag-tiger.py" --version || exit /b 1
)

if /I not "%VIRTUAL_ENV%" == "%~dp0.venv" (
    echo Activating virtual environment...
    call "%~dp0.venv\Scripts\activate.bat"
)

python "%~dp0ragtag-tiger.py" %*
