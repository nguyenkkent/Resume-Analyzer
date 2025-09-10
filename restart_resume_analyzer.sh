#!/bin/bash
# restart_resume_analyzer.sh
# One-shot restart for Resume Analyzer (PyTorch + FastAPI + kind)

echo "1️⃣ Start Docker Desktop and make sure Docker is running"
docker --version || { echo "Docker not running! Start Docker Desktop."; exit 1; }

echo "2️⃣ Check or create kind cluster"
kind get clusters | grep resume-ai >/dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Creating kind cluster 'resume-ai'..."
  kind create cluster --name resume-ai
else
  echo "Kind cluster 'resume-ai' already exists"
fi

echo "3️⃣ Build Docker image"
cd ~/Documents/Projects/Resume-Analyzer/services/embedder-pyTorch
docker build -t resume-analyzer .

echo "4️⃣ Load Docker image into kind"
kind load docker-image resume-analyzer --name resume-ai

echo "5️⃣ Apply Kubernetes manifests"
cd ~/Documents/Projects/Resume-Analyzer/k8s
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

echo "6️⃣ Delete old pods to refresh image"
kubectl delete pod -l app=resume-analyzer >/dev/null 2>&1 || true

echo "7️⃣ Wait for Pod to be ready"
echo "Watching pods (CTRL+C to stop watching)..."
kubectl get pods -w

echo "✅ Done! Once pod shows '1/1 Ready', run:"
echo "kubectl port-forward deployment/resume-analyzer 8000:8000"
echo "Then open Swagger UI at http://localhost:8000/docs or test API with curl"
