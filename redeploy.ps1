# Rebuild, load into Kind, and restart the deployment (fast inner loop)

$ErrorActionPreference = "Stop"
Write-Host "Building image..."
docker build -t resume-analyzer -f ./services/embedder-pyTorch/Dockerfile .

Write-Host "Loading image into Kind..."
kind load docker-image resume-analyzer --name resume-ai

Write-Host "Restarting deployment..."
kubectl rollout restart deployment resume-analyzer

Write-Host "`n✅ Redeploy triggered. Check pod:"
kubectl get pods -l app=resume-analyzer
