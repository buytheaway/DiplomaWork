$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location -Path ".." | Out-Null

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example (EMBEDDING_BACKEND=dummy) ..."
    Copy-Item ".env.example" ".env"
}

Write-Host "Starting Docker Compose ..."
docker compose up --build

Write-Host ""
Write-Host "Health:  http://localhost:8000/v1/health"
Write-Host "Docs:    http://localhost:8000/docs"
Write-Host "Stats:   http://localhost:8000/v1/index/stats"
