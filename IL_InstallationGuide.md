# Installation Manual
## Agentic Trip Planning System

---

## Overview

This installation manual provides step-by-step instructions to set up and run the Agentic Trip Planning System from a fresh installation.

**System Type:** Multi-Agent AI Trip Planner with ReAct Reasoning
**Technology Stack:** FastAPI (Backend) + HTML/CSS/JS (Frontend) + Ollama (Local LLM)

---

## Prerequisites

### Required Software

1. **Python 3.9 or higher**
   - Download: https://www.python.org/downloads/
   - Verify installation: `python --version`

2. **Ollama (Local LLM Server)**
   - Download: https://ollama.ai/
   - Installation:
     - Windows: Download and run the installer from ollama.ai
     - macOS: `brew install ollama`
     - Linux: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Verify installation: `ollama --version`

3. **Git** (if cloning from repository)
   - Download: https://git-scm.com/downloads

---

## Installation Steps

### 1. Extract/Clone the Project

If you received the source code as a ZIP file:
```bash
# Extract to your desired location
cd C:\Users\YourName\Downloads\project\research-project
```

If cloning from a repository:
```bash
git clone <repository-url>
cd research-project
```

### 2. Install Ollama Model

The system requires the `mistral-small` model:

```bash
# Pull the mistral-small model
ollama pull mistral-small

# Verify the model is installed
ollama list
```

**Important:** The model download is approximately 4.1GB. Ensure you have sufficient disk space and a stable internet connection.

### 3. Set Up Python Virtual Environment

**Windows:**
```bash
# Navigate to project directory
cd C:\Users\YourName\Downloads\project\research-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**macOS/Linux:**
```bash
# Navigate to project directory
cd ~/Downloads/project/research-project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal prompt.

### 4. Install Python Dependencies

With the virtual environment activated:

```bash
pip install fastapi uvicorn pydantic httpx langchain-ollama langgraph
```

**Core Dependencies:**
- `fastapi` - Web framework for backend API
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `httpx` - HTTP client for routing API
- `langchain-ollama` - Ollama LLM integration
- `langgraph` - Multi-agent workflow orchestration

### 5. Verify Installation

Check that all components are installed correctly:

```bash
# Check Python packages
pip list | grep -E "fastapi|uvicorn|langchain|langgraph"

# Check Ollama is running
ollama list
```

---

## Running the Application

### 1. Start Ollama Service

**Windows:**
- Ollama runs as a background service automatically after installation
- Verify it's running by opening http://localhost:11434 in a browser

**macOS/Linux:**
```bash
# Start Ollama service (if not running)
ollama serve
```

### 2. Start the Backend Server

Open a terminal in the project directory:

```bash
# Activate virtual environment (if not already active)
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Navigate to backend directory
cd backend

# Start the FastAPI server
python main.py
```

You should see:
```
🚀 Starting Agentic Trip Planner API...
📖 API Documentation: http://localhost:8000/docs
🌐 Frontend: http://localhost:8000
📊 Available cities: NYC, BOS, CHI, SF
```

### 3. Access the Application

Open your web browser and navigate to:

**Frontend Interface:**
http://localhost:8000

**API Documentation:**
http://localhost:8000/docs

**Health Check:**
http://localhost:8000/health

---

## Configuration

### Environment Variables

The system does not require API keys or cloud services. All AI processing is done locally via Ollama.

**Optional Configuration:**
- **Model Selection:** Default is `mistral-small`. To use a different model, modify `backend/agents/base_agent.py:83`
- **Port:** Default is 8000. To change, modify `backend/main.py:377`

### Available Cities (Demo Data)

The proof-of-concept includes data for:
- **NYC** - New York City
- **BOS** - Boston
- **CHI** - Chicago
- **SF** - San Francisco

Flights and hotels are pre-generated for these cities only.

---

## Troubleshooting

### Issue: "Model not found" error
**Solution:** Ensure mistral-small is installed:
```bash
ollama pull mistral-small
ollama list
```

### Issue: Port 8000 already in use
**Solution:** Kill the process using port 8000 or change the port in `main.py`
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

### Issue: "No module named 'fastapi'"
**Solution:** Ensure virtual environment is activated and dependencies are installed:
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt  # If available
```

### Issue: Ollama connection refused
**Solution:** Ensure Ollama service is running:
```bash
ollama serve  # Start Ollama service
```

### Issue: Frontend shows 404 or doesn't load
**Solution:** Ensure you're running `python main.py` from the `backend` directory, not the project root.

---

## Minimum System Requirements

- **CPU:** 4 cores recommended (for local LLM inference)
- **RAM:** 8GB minimum, 16GB recommended
- **Disk Space:** 10GB (includes Ollama model)
- **OS:** Windows 10/11, macOS 11+, or modern Linux distribution
- **Browser:** Chrome, Firefox, Edge, or Safari (latest versions)

---

## Cloud Setup

**No cloud resources required.** This system runs entirely locally:
- No Azure Resource Groups needed
- No API keys required
- No external AI services (uses local Ollama)

The only external service used is the OSRM routing API (http://router.project-osrm.org) for calculating hotel distances, which is a free public service.

---

## Next Steps

Once installation is complete, refer to the **User Manual** for instructions on using the application.

---

**Version:** 1.0.0
**Last Updated:** 2026-01-24
