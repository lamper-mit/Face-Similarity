#!/bin/bash

# Step 1: Create a virtual environment named 'venv'
python3 -m venv venv

# Step 2: Activate the virtual environment
source venv/bin/activate

# Step 3: Install dependencies from requirements.txt
pip install -r requirements_linux.txt

# Step 4: Make the main script (compare.py) executable
chmod +x compare_linux.py

echo "Setup completed. Virtual environment created and dependencies installed."
echo "To activate the virtual environment, use: source venv/bin/activate"
