#!/bin/bash
# AI Data Platform 2025 - Linux/Mac Startup Script
# Run: chmod +x start.sh && ./start.sh

echo "============================================================"
echo "   AI Data Platform 2025 - Starting..."
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"
echo ""

# Start the application
echo "Starting the application..."
echo ""
python3 modern_ui_complete.py

# If the script exits with error
if [ $? -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "   Application stopped with an error"
    echo "============================================================"
    read -p "Press Enter to exit..."
fi
