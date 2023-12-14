@echo off
REM Step 1: Create a virtual environment named 'venv'
python -m venv venv

REM Step 2: Activate the virtual environment
call venv\Scripts\activate

REM Step 3: Install dependencies from requirements.txt
pip install -r requirements_windows.txt

echo Setup completed. Virtual environment created and dependencies installed.
echo To activate the virtual environment, use: .\venv\Scripts\activate

