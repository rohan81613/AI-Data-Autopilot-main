#!/usr/bin/env python3
"""
Installation Verification Script
Checks if all required packages are installed correctly
"""
import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - NOT INSTALLED")
        return False

def main():
    print("=" * 60)
    print("🔍 AI Data Platform 2025 - Installation Verification")
    print("=" * 60)
    print()
    
    packages = [
        ("gradio", "gradio"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("plotly", "plotly"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("statsmodels", "statsmodels"),
        ("reportlab", "reportlab"),
        ("python-pptx", "pptx"),
        ("llama-cpp-python", "llama_cpp"),
        ("psutil", "psutil"),
        ("openpyxl", "openpyxl"),
        ("Pillow", "PIL"),
        ("joblib", "joblib"),
    ]
    
    print("Checking required packages:\n")
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    print()
    print("=" * 60)
    
    if all_installed:
        print("✅ All packages installed successfully!")
        print()
        print("You can now run the application:")
        print("  python run.py")
        print("  or")
        print("  python modern_ui_complete.py")
    else:
        print("❌ Some packages are missing!")
        print()
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return 0 if all_installed else 1

if __name__ == "__main__":
    sys.exit(main())
