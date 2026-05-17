param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Quote-Ps {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Read-DotEnv {
    param([string]$Path)

    $values = @{}
    if (-not (Test-Path $Path)) {
        throw "Missing env file: $Path"
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or $line -notmatch "=") {
            return
        }
        $key, $value = $line -split "=", 2
        $values[$key.Trim()] = $value.Trim()
    }
    return $values
}

function Wait-Docker {
    Write-Step "Checking Docker"
    try {
        docker info *> $null
        Write-Host "Docker is already running." -ForegroundColor Green
        return
    } catch {
        $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (-not (Test-Path $dockerDesktop)) {
            throw "Docker is not running and Docker Desktop was not found at $dockerDesktop"
        }
        Write-Host "Starting Docker Desktop..." -ForegroundColor Yellow
        Start-Process -FilePath $dockerDesktop
    }

    for ($i = 1; $i -le 90; $i++) {
        Start-Sleep -Seconds 2
        try {
            docker info *> $null
            Write-Host "Docker is ready." -ForegroundColor Green
            return
        } catch {
            if ($i % 10 -eq 0) {
                Write-Host "Waiting for Docker... $($i * 2)s"
            }
        }
    }
    throw "Docker did not become ready within 180 seconds."
}

function Ensure-ContainerRunning {
    param([string]$ContainerName)

    Write-Step "Starting PostgreSQL container"
    $exists = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }
    if (-not $exists) {
        throw "Required container '$ContainerName' was not found. Create/restore the scale PostgreSQL container first."
    }

    docker start $ContainerName *> $null
    for ($i = 1; $i -le 30; $i++) {
        Start-Sleep -Seconds 1
        $running = docker inspect -f "{{.State.Running}}" $ContainerName
        if ($running -eq "true") {
            Write-Host "PostgreSQL container is running: $ContainerName" -ForegroundColor Green
            return
        }
    }
    throw "Container '$ContainerName' did not start."
}

function Test-Backend {
    param([hashtable]$EnvValues)

    $headers = @{}
    if ($EnvValues["API_KEY"]) {
        $headers["X-API-Key"] = $EnvValues["API_KEY"]
    }

    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/health" -Headers $headers -TimeoutSec 3
        return $health.status -eq "ok"
    } catch {
        return $false
    }
}

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$envFile = Join-Path $repo ".env.docker"
$envValues = Read-DotEnv $envFile

$containerName = "diplomawork_scale_pg_10k_test"
$scaleDbName = "biometric_scale_custom_1m_templates"
$postgresUser = "scale_user"
$backendLog = Join-Path $repo "tmp\full_demo_backend.log"
$desktopLog = Join-Path $repo "tmp\full_demo_desktop.log"
$backendRuntimeScript = Join-Path $repo "tmp\start_backend_full_demo.ps1"
$desktopRuntimeScript = Join-Path $repo "tmp\start_desktop_full_demo.ps1"

New-Item -ItemType Directory -Force (Join-Path $repo "tmp") | Out-Null

Wait-Docker
Ensure-ContainerRunning $containerName

Write-Step "Checking scale database"
$postgresPassword = docker exec $containerName printenv POSTGRES_PASSWORD
if (-not $postgresPassword) {
    throw "Could not read POSTGRES_PASSWORD from '$containerName'."
}
docker exec $containerName psql -U $postgresUser -d $scaleDbName -t -A -c "select 1;" *> $null
$databaseUrl = "postgresql+psycopg2://$postgresUser`:$postgresPassword@127.0.0.1:55432/$scaleDbName"
Write-Host "Scale database is reachable: $scaleDbName" -ForegroundColor Green

Write-Step "Checking runtime artifacts"
$requiredPaths = @(
    "models\det_10g.onnx",
    "models\w600k_r50.onnx",
    "diplomcheckbackup\training\outputs_medium_lfw_finetune\best_lfw.pth",
    "diplomcheckbackup\training\detector_runs\face_yolo_ft1\weights\best.pt",
    "tmp\scale_pretrained_2m_combined",
    "tmp\scale_custom_2m_synced"
)
foreach ($relative in $requiredPaths) {
    $path = Join-Path $repo $relative
    if (-not (Test-Path $path)) {
        throw "Missing runtime artifact/path: $relative"
    }
}
Write-Host "Runtime artifacts are present." -ForegroundColor Green

if ($CheckOnly) {
    Write-Step "CheckOnly finished"
    exit 0
}

Write-Step "Starting backend window"
if (Test-Backend $envValues) {
    Write-Host "Backend is already responding on http://127.0.0.1:8000" -ForegroundColor Green
} else {
    $backendLines = @(
        '$Host.UI.RawUI.WindowTitle = "DiplomaWork Backend"',
        '$ErrorActionPreference = "Stop"',
        ('$env:DATABASE_URL = ' + (Quote-Ps $databaseUrl)),
        ('$env:API_KEY = ' + (Quote-Ps $envValues["API_KEY"])),
        ('$env:ADMIN_API_KEY = ' + (Quote-Ps $envValues["ADMIN_API_KEY"])),
        ('$env:DATA_ENCRYPTION_KEY = ' + (Quote-Ps $envValues["DATA_ENCRYPTION_KEY"])),
        ('$env:SNAPSHOT_ENCRYPTION_KEY = ' + (Quote-Ps $envValues["SNAPSHOT_ENCRYPTION_KEY"])),
        '$env:DEFAULT_PIPELINE = "custom"',
        '$env:ENABLE_PRETRAINED_PIPELINE = "true"',
        '$env:ENABLE_CUSTOM_PIPELINE = "true"',
        '$env:PRETRAINED_BACKEND = "onnx"',
        '$env:EMBEDDING_BACKEND = "onnx"',
        '$env:DETECTION_BACKEND = "none"',
        ('$env:ONNX_DETECTOR_PATH = ' + (Quote-Ps (Join-Path $repo "models\det_10g.onnx"))),
        ('$env:ONNX_EMBEDDER_PATH = ' + (Quote-Ps (Join-Path $repo "models\w600k_r50.onnx"))),
        '$env:CUSTOM_BACKEND = "torch"',
        '$env:CUSTOM_DETECTION_BACKEND = "yolo"',
        '$env:CUSTOM_MIN_DET_SCORE = "0.22"',
        '$env:CUSTOM_FACE_CROP_MARGIN = "0.30"',
        '$env:CUSTOM_YOLO_IMGSZ = "1280"',
        ('$env:TORCH_MODEL_PATH = ' + (Quote-Ps (Join-Path $repo "diplomcheckbackup\training\outputs_medium_lfw_finetune\best_lfw.pth"))),
        '$env:TORCH_MODEL_ARCH = "ir50"',
        '$env:TORCH_DEVICE = "cuda"',
        '$env:TORCH_USE_FP16 = "true"',
        ('$env:YOLO_MODEL_PATH = ' + (Quote-Ps (Join-Path $repo "diplomcheckbackup\training\detector_runs\face_yolo_ft1\weights\best.pt"))),
        ('$env:PRETRAINED_INDEX_PATH = ' + (Quote-Ps (Join-Path $repo "tmp\scale_pretrained_2m_combined\pretrained.faiss"))),
        ('$env:CUSTOM_INDEX_PATH = ' + (Quote-Ps (Join-Path $repo "tmp\scale_custom_2m_synced\custom.faiss"))),
        '$env:INDEX_TYPE = "ivfpq"',
        '$env:IVFPQ_NLIST = "4096"',
        '$env:IVFPQ_M = "32"',
        '$env:IVFPQ_NBITS = "8"',
        '$env:IVFPQ_NPROBE = "32"',
        '$env:IVFPQ_NPROBE_FAST = "128"',
        '$env:IVFPQ_NPROBE_SAFE = "128"',
        '$env:SEARCH_CANDIDATE_K = "200"',
        '$env:SEARCH_CANDIDATE_K_FAST = "500"',
        '$env:SEARCH_CANDIDATE_K_SAFE = "500"',
        '$env:SEARCH_DYNAMIC_ENABLED = "true"',
        '$env:SEARCH_FALLBACK_MARGIN = "0.05"',
        '$env:MATCH_THRESHOLD = "0.4"',
        '$env:PRETRAINED_MATCH_THRESHOLD = "0.4"',
        '$env:CUSTOM_MATCH_THRESHOLD = "0.30"',
        '$env:CUSTOM_LIVE_MATCH_THRESHOLD = "0.30"',
        '$env:RATE_LIMIT_ENABLED = "false"',
        '$env:AUTO_SAVE_INDEX = "false"',
        ('Set-Location ' + (Quote-Ps (Join-Path $repo "backend"))),
        '$ErrorActionPreference = "Continue"',
        'python -m uvicorn app.main:app --host 127.0.0.1 --port 8000'
    )
    Set-Content -LiteralPath $backendRuntimeScript -Value $backendLines -Encoding UTF8

    Start-Process powershell -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $backendRuntimeScript
    )

    for ($i = 1; $i -le 45; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Backend $envValues) {
            Write-Host "Backend is ready: http://127.0.0.1:8000" -ForegroundColor Green
            break
        }
        if ($i -eq 45) {
            Write-Host "Backend did not answer yet. Check the 'DiplomaWork Backend' window or logs:" -ForegroundColor Yellow
            Write-Host "  $backendLog"
        }
    }
}

Write-Step "Starting desktop window"
$desktopLines = @(
    '$Host.UI.RawUI.WindowTitle = "DiplomaWork Desktop"',
    'function Stop-DemoBackend {',
    '    $needle = "uvicorn " + "app.main:app"',
    '    Get-CimInstance Win32_Process -Filter "name = ''python.exe''" |',
    '        Where-Object { $_.CommandLine -like "*$needle*" } |',
    '        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }',
    '    Get-Process powershell,pwsh -ErrorAction SilentlyContinue |',
    '        Where-Object { $_.MainWindowTitle -eq "DiplomaWork Backend" } |',
    '        Stop-Process -Force -ErrorAction SilentlyContinue',
    '}',
    ('$env:API_BASE_URL = ' + (Quote-Ps "http://127.0.0.1:8000")),
    ('$env:API_KEY = ' + (Quote-Ps $envValues["API_KEY"])),
    ('$env:ADMIN_API_KEY = ' + (Quote-Ps $envValues["ADMIN_API_KEY"])),
    '$env:CAMERA_FRAME_WIDTH = "1280"',
    '$env:CAMERA_FRAME_HEIGHT = "720"',
    '$env:LIVE_SCAN_INTERVAL_MS = "150"',
    '$env:LIVE_MAX_WIDTH = "640"',
    '$env:LIVE_JPEG_QUALITY = "72"',
    '$env:CUSTOM_LIVE_MAX_WIDTH = "1280"',
    '$env:CUSTOM_LIVE_JPEG_QUALITY = "82"',
    ('Set-Location ' + (Quote-Ps (Join-Path $repo "desktop"))),
    '$ErrorActionPreference = "Continue"',
    'try {',
    '    python -m app.main',
    '} finally {',
    '    Stop-DemoBackend',
    '}'
) 
Set-Content -LiteralPath $desktopRuntimeScript -Value $desktopLines -Encoding UTF8

Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $desktopRuntimeScript
)

Write-Step "Launch complete"
Write-Host "Opened windows:" -ForegroundColor Green
Write-Host "  - Docker Desktop / PostgreSQL container"
Write-Host "  - DiplomaWork Backend"
Write-Host "  - DiplomaWork Desktop"
Write-Host ""
Write-Host "Backend logs:"
Write-Host "  $backendLog"
Write-Host "Desktop logs:"
Write-Host "  $desktopLog"
