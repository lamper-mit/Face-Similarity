#!/usr/bin/env python
import venv
import subprocess
import sys

def main():
    # Create a virtual environment named 'venv'
    venv.create('venv', with_pip=True)

    # Install dependencies from requirements.txt within the virtual environment
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

if __name__ == "__main__":
    main()
