Write-Host "Stopping Resume Analyzer..."

# Stop port forwarding (kill any process using port 8000)
$port = 8000
$proc = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($proc) {
    Write-Host "Stopping process on port $port..."
    Stop-Process -Id $proc -Force
}

# Delete pods
kubectl delete pod -l app=resume-analyzer --force --grace-period=0

# Delete Kind cluster
Write-Host "Deleting Kind cluster..."
kind delete cluster --name resume-ai

Write-Host "`n✅ Teardown complete."
