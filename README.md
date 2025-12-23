# 🚀 AI Data Platform 2025

**A complete, professional-grade AI-powered data analysis platform with one-click ML, AI assistant, and automated reporting**

[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-orange)]()
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)]()

---

## ⚠️ IMPORTANT: AI Model Not Included

**The AI Assistant LLM model is NOT included in this repository** (it's 2.5 GB).

**You need to download it separately:**
- **Download Link:** [Phi-3.5-mini-instruct-Q5_K_M.gguf](https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q5_K_M.gguf) (~2.5 GB)
- **Place in:** `models/Phi-3.5-mini-instruct-Q5_K_M.gguf`
- **Instructions:** See [models/DOWNLOAD_MODEL.md](models/DOWNLOAD_MODEL.md)

> **Note:** The application works perfectly without the AI model! All features (data cleaning, ML, visualization, reports) function normally. Only the AI Chat Assistant requires the model.

---

## ⚡ Quick Start

### 3 Simple Steps:

```bash
# 1. Clone the repository
git clone https://github.com/PatelJU/AI-Data-Autopilot.git
cd AI-Data-Autopilot

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Run the application
python modern_ui_complete.py
```

**Or use the startup scripts:**
- **Windows**: Double-click `start.bat`
- **Linux/Mac**: Run `./start.sh`

**Open in browser:** http://127.0.0.1:7864

> 📖 **Need help?** See [HOW_TO_RUN.md](HOW_TO_RUN.md) for detailed instructions!

### 🤖 Optional: Enable AI Assistant

To use the AI Chat Assistant feature:
1. Download the model: [Phi-3.5-mini-instruct-Q5_K_M.gguf](https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q5_K_M.gguf) (~2.5 GB)
2. Place it in: `models/Phi-3.5-mini-instruct-Q5_K_M.gguf`
3. Restart the application

See [models/DOWNLOAD_MODEL.md](models/DOWNLOAD_MODEL.md) for detailed instructions.

---

## ✨ Key Features

- 🤖 **AI Assistant** - Chat with Phi-3.5 LLM
- 🚀 **Smart Autopilot** - One-click cleaning & ML
- 📊 **5 ML Algorithms** - XGBoost, LightGBM, Random Forest, etc.
- 📄 **PDF Reports** - Professional A4-sized reports
- 📊 **PowerPoint** - Quick & AI-powered presentations
- 📈 **Smart Dashboard** - Auto-generates best visualizations
- 🔄 **21 Auto-Synced Dropdowns** - Everything stays updated

---

## 📖 Documentation

**For complete documentation, see:** [PROJECT_README.md](PROJECT_README.md)

Includes:
- Installation guide
- Feature documentation
- How-to guides
- Troubleshooting
- Best practices
- Technical details

---

## 🎯 Perfect For

- 🎓 **Students** - Learn data science without coding
- 💼 **Professionals** - Quick analysis & reports
- 🔬 **Researchers** - Statistical analysis & ML
- 🏢 **Businesses** - Data-driven decisions

---

## 🚀 What Makes It Special

1. **Beginner-Friendly** - One-click operations
2. **Professional-Grade** - Enterprise-quality output
3. **AI-Powered** - LLM integration for intelligence
4. **Complete Workflow** - Upload → Clean → Analyze → Model → Export
5. **Free & Local** - No cloud, no cost, full privacy

---

## 📊 Quick Example

```python
# 1. Start platform
python modern_ui_complete.py

# 2. Upload data (via UI)
# 3. Click "One-Click Clean"
# 4. Click "One-Click ML"
# 5. Generate PDF report
# Done! Professional analysis in minutes!
```

---

## 🛠️ Requirements

- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended for AI features)
- **OS**: Windows, Linux, or macOS
- **Internet**: Required for initial package installation

All dependencies are listed in `requirements.txt`.

### 🤖 AI Assistant Setup (Optional)

The AI Assistant requires a local LLM model:

**Download:** [Phi-3.5-mini-instruct-Q5_K_M.gguf](https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q5_K_M.gguf) (~2.5 GB)

**Place in:** `models/Phi-3.5-mini-instruct-Q5_K_M.gguf`

📖 **Detailed instructions:** See [models/DOWNLOAD_MODEL.md](models/DOWNLOAD_MODEL.md)

> **Note:** The application works perfectly without the AI model! All features (ML, visualization, reports) function normally. Only the AI Chat Assistant requires the model.

---

## 📁 Project Structure

```
├── run.py                   # Quick start script (recommended)
├── modern_ui_complete.py    # Main application
├── requirements.txt         # All dependencies
├── ai/                      # AI Assistant & LLM
├── utils/                   # Utilities (reports, export, etc.)
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── models/                  # LLM model files
├── reports/                 # Generated PDF reports
├── presentations/           # Generated PowerPoint files
├── optional/                # Advanced features (not required)
└── PROJECT_README.md        # Complete documentation
```

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for running instructions

---

## 🎉 Status

✅ **Production Ready**  
✅ **All Features Working**  
✅ **Fully Documented**  
✅ **Professional Grade**

---

## 📞 Support

- 📖 Read [PROJECT_README.md](PROJECT_README.md) for complete guide
- 🐛 Check troubleshooting section
- 💡 Use Help tabs in the application

---

## 🏆 Credits

Built with: Gradio, Plotly, scikit-learn, XGBoost, LightGBM, Phi-3.5

---

**Start your AI-powered data analysis journey today!** 🚀

```powershell
python modern_ui_complete.py
```
