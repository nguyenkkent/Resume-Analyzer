# 🧠 **Resumi AI (FastAPI + PyTorch + Gemini + Kind)**

A local, container-ready AI resume analyzer built with **FastAPI**, **PyTorch**, and **LangChain**—deployable via Docker and Kubernetes (**Kind**).  
It compares resumes to job descriptions, computes similarity scores, and suggests missing skills using embeddings.

---

## 🚀 **Quick Start (Windows 11)**

### 1. Clone the Repository
```powershell
git clone https://github.com/<your-username>/resume-analyzer.git
cd resume-analyzer
```

---

### 2. Create Virtual Environments

This project uses **two** Python virtual environments:
- `venv` → Agent (Python 3.14)  
- `venv312` → Embedder (Python 3.12)

```powershell
# --- Agent environment (Python 3.14) ---
# Used to run: services/agent/agent.py
python -m venv venv
.env\Scriptsctivate

python --version    # Should show 3.14.x
pip install -r requirements.txt
deactivate


# --- Embedder environment (Python 3.12) ---
# Used to run: services/embedder-pyTorch/app.py
python3.12 -m venv venv312
.env312\Scriptsctivate

python --version    # Should show 3.12.x
pip install -r services/embedder-pyTorch/requirements.txt
deactivate
```

---

### 3. Install Dependencies (for Agent)
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Environment Variables

Create a `.env` file in the project root:

```
# API keys for LLM and Search engine
TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_gemini_key

# Where the agent can reach the embedder
EMBEDDER_URL=http://localhost:8000

# Optional: turn on verbose logs in the agent
DEBUG_AGENT=1
```

---

### 5. Run Everything (Three Terminals)

#### 🧩 Terminal A — Agent (FastAPI on port 8100)
```powershell
.env\Scriptsctivate
$env:DEBUG_AGENT="1"
uvicorn services.agent.agent:app --reload --port 8100
# Swagger: http://localhost:8100/docs
```

#### 🔢 Terminal B — Embedder (FastAPI on port 8000)
```powershell
.env312\Scriptsctivate
cd .\services\embedder-pyTorch
uvicorn app:app --reload --port 8000
# Swagger: http://localhost:8000/docs
```

#### 💬 Terminal C — Frontend (Vite + Tailwind)
```powershell
cd .
esumi-ai
npm install
npm run dev
# Opens http://localhost:5173
```

---

### ⚙️ Troubleshooting

**Agent 502 / JSON parse errors**  
→ Ensure `EMBEDDER_URL=http://localhost:8000` and the embedder service is running.

**Model not found (Gemini)**  
→ Use a verified model from your list-models script (e.g., `gemini-2.5-flash`). Free tier limits apply.

**CORS in browser**  
→ CORS is enabled for `http://localhost:*`. If you still see errors, confirm both FastAPI servers are running and the Vite proxy is active.

---

### 🧰 Tech Stack

| Layer | Technology |
|:------|:------------|
| **Backend API** | FastAPI |
| **Embeddings Model** | PyTorch (Sentence Transformers) |
| **LLM Agent** | Google Gemini |
| **Search Engine** | Tavily |
| **Frontend UI** | React + Vite + Tailwind CSS |
| **Containerization** | Docker |
| **Local Orchestration** | Kind (Kubernetes in Docker) |
| **Dev Environment** | VS Code |
