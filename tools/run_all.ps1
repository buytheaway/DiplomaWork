$ErrorActionPreference = "Stop"

Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location -Path ".." | Out-Null

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

docker compose up --build

Write-Host ""
Write-Host "Health: http://localhost:8000/v1/health"
Write-Host "Docs:   http://localhost:8000/docs"
