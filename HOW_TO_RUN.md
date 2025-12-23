# 🚀 How to Run AI Data Platform 2025

## Quick Start (3 Steps)

### Step 1: Install Python
Make sure you have Python 3.8 or higher installed.
```bash
python --version
```

### Step 2: Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

#### Windows Users:
**Easiest way:** Double-click `start.bat`

Or in terminal:
```bash
start.bat
```

#### Linux/Mac Users:
```bash
chmod +x start.sh
./start.sh
```

#### All Platforms (Direct Python):
```bash
python modern_ui_complete.py
```

### Step 4: Open Browser
The application will automatically start at:
```
http://127.0.0.1:7864
```

## Troubleshooting

### Issue: "Python not found"
**Solution:** Install Python from https://python.org (version 3.8 or higher)

### Issue: "Module not found" errors
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Port 7864 already in use
**Solution:** Edit `modern_ui_complete.py` and change the port:
```python
demo.launch(server_port=7865)  # Change to any available port
```

### Issue: Application starts but stops immediately
**Solution:** Check the error message in the terminal. Usually it's a missing dependency.

### Issue: LLM model not found
**Solution:** Download the AI model:
- **Link:** https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q5_K_M.gguf
- **Place in:** `models/Phi-3.5-mini-instruct-Q5_K_M.gguf`
- **Guide:** See [models/DOWNLOAD_MODEL.md](models/DOWNLOAD_MODEL.md)

**Note:** App works without the model! Only AI Chat needs it.

## Verify Installation

Before running, you can verify all packages are installed:
```bash
python verify_installation.py
```

## Stopping the Application

Press `Ctrl+C` in the terminal where the application is running.

## Need More Help?

- See [PROJECT_README.md](PROJECT_README.md) for complete documentation
- Check the Troubleshooting section above for common issues

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended for AI features)
- **Disk Space**: 5 GB (including model files)
- **OS**: Windows, Linux, or macOS

## What Happens When You Run?

1. ✅ Loads all Python modules
2. ✅ Initializes AI Assistant (if model available)
3. ✅ Creates the web interface
4. ✅ Starts the server on port 7864
5. ✅ Ready to use!

The entire startup takes about 10-30 seconds depending on your system.

---

**That's it! Enjoy your AI Data Platform!** 🎉
