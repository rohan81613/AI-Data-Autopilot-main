#!/usr/bin/env python3
"""
AI Data Platform 2025 - Quick Start Script
Run this file to start the application
"""
import subprocess
import sys
import os

def check_requirements():
    """Check if requirements are installed"""
    try:
        import gradio
        import pandas
        import plotly
        import sklearn
        return True
    except ImportError:
        return False

def install_requirements():
    """Install requirements automatically"""
    print("📦 Installing required packages...")
    print("This may take a few minutes on first run.\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("\n✅ All packages installed successfully!\n")

def main():
    print("=" * 60)
    print("🚀 AI Data Platform 2025")
    print("=" * 60)
    print()
    
    # Check if requirements are installed
    if not check_requirements():
        print("⚠️  Required packages not found.")
        response = input("Would you like to install them now? (y/n): ")
        if response.lower() == 'y':
            try:
                install_requirements()
            except Exception as e:
                print(f"\n❌ Error installing packages: {e}")
                print("Please run manually: pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("\n❌ Cannot start without required packages.")
            print("Please run: pip install -r requirements.txt")
            sys.exit(1)
    
    # Start the application
    print("🎯 Starting AI Data Platform...")
    print("📍 The application will open in your browser")
    print("🌐 URL: http://127.0.0.1:7864")
    print("\n💡 Press Ctrl+C to stop the server\n")
    print("=" * 60)
    print()
    
    try:
        # Run the main application directly
        subprocess.run([sys.executable, "modern_ui_complete.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        print("Thank you for using AI Data Platform 2025!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting application (exit code {e.returncode})")
        print("\nPlease check:")
        print("1. All requirements are installed: pip install -r requirements.txt")
        print("2. Python version is 3.8 or higher")
        print("3. All project files are present")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
