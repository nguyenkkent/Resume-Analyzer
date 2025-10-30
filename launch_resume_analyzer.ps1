# Launch Resume Analyzer (Windows, full setup)
# Rebuilds venv if needed, installs requirements, builds Docker image, deploys via Kind

$ErrorActionPreference = "Stop"

# 1️⃣ Create or activate virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "🧱 Creating Python virtual environment..."
    python -m venv venv
}

Write-Host "🚀 Activating virtual environment..."
. .\venv\Scripts\Activate.ps1

# 2️⃣ Ensure dependencies are installed
if (-not (Test-Path "venv\Lib\site-packages\fastapi")) {
    Write-Host "📦 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
} else {
    Write-Host "📦 Dependencies already installed."
}

# 3️⃣ Ensure Docker is running
Write-Host "🐋 Checking Docker..."
try {
    docker version | Out-Null
} catch {
    Write-Host "❌ Docker not running. Please start Docker Desktop first."
    exit 1
}

# 4️⃣ Build Docker image
Write-Host "🔧 Building Docker image..."
docker build -t resume-analyzer ./services/embedder-pyTorch

# 5️⃣ Create Kind cluster if not exists
if (-not (kind get clusters | Select-String "resume-ai")) {
    Write-Host "🌱 Creating Kind cluster 'resume-ai'..."
    kind create cluster --name resume-ai
} else {
    Write-Host "🌱 Kind cluster 'resume-ai' already exists."
}

# 6️⃣ Load Docker image into Kind
Write-Host "📦 Loading image into Kind cluster..."
kind load docker-image resume-analyzer --name resume-ai

# 7️⃣ Apply Kubernetes manifests
Write-Host "⚙️ Deploying to Kubernetes..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 8️⃣ Wait a few seconds for pod creation
Start-Sleep -Seconds 5

# 9️⃣ Port-forward service
Write-Host "🔌 Port-forwarding to localhost:8000..."
Start-Process powershell -ArgumentList "kubectl port-forward deployment/resume-analyzer 8000:8000"

Write-Host "`n✅ Resume Analyzer is running!"
Write-Host "Visit: http://localhost:8000/docs"
