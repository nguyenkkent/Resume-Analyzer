# 🧠 Resume Analyzer (FastAPI + PyTorch + LangChain + Kind)

A containerized AI resume analyzer built with FastAPI, PyTorch, and LangChain — deployed locally via Docker and Kubernetes (Kind).  
It compares resumes to job descriptions, scores relevance, and suggests missing skills using embeddings.

---

## 🚀 Quick Start (Windows 11)

### 1. Clone the Repo
git clone https://github.com/<your-username>/resume-analyzer.git
cd resume-analyzer


### 2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate


### 3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt


### 4. Build the Docker Image
docker build -t resume-analyzer ./services/embedder-pyTorch


### 5. Create & Load into Kind Cluster
kind create cluster --name resume-ai
kind load docker-image resume-analyzer --name resume-ai


### 6. Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

### 7. Port-Forward to Access API
kubectl port-forward deployment/resume-analyzer 8000:8000


### Cleanup
kubectl delete pod -l app=resume-analyzer --force --grace-period=0
kind delete cluster --name resume-ai


### Tech Stack:
FastAPI — Backend API
PyTorch — Embedding model
LangChain + Tavily — Agent integration (in progress)
Docker — Containerization
Kind (Kubernetes in Docker) — Local orchestration
VS Code — Dev environment