$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location -Path ".." | Out-Null

if (-not (Test-Path ".env.docker")) {
    Write-Host "Creating .env.docker from .env.docker.example ..."
    Copy-Item ".env.docker.example" ".env.docker"
}

Write-Host "Starting Docker Compose (db + backend dual runtime) ..."
docker compose up --build

Write-Host ""
Write-Host "Health:  http://localhost:8000/v1/health"
Write-Host "Docs:    http://localhost:8000/docs"
Write-Host "Stats:   http://localhost:8000/v1/index/stats"
