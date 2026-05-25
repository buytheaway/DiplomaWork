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
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing env file: $Path"
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or $line -notmatch "=") {
            return
        }
        $key, $value = $line -split "=", 2
        $values[$key.Trim()] = $value.Trim().Trim('"')
    }
    return $values
}

function Add-EnvLines {
    param(
        [System.Collections.IDictionary]$Values,
        [System.Collections.Generic.List[string]]$Lines
    )

    foreach ($key in ($Values.Keys | Sort-Object)) {
        if (-not $key) {
            continue
        }
        $Lines.Add(('$env:{0} = {1}' -f $key, (Quote-Ps ([string]$Values[$key]))))
    }
}

function Wait-Docker {
    Write-Step "Checking Docker"
    try {
        docker info *> $null
        Write-Host "Docker is ready." -ForegroundColor Green
        return
    } catch {
        $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (-not (Test-Path -LiteralPath $dockerDesktop)) {
            throw "Docker is not running and Docker Desktop was not found."
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

function Stop-BackendOnPort {
    param([int]$Port)

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        if ($connection.OwningProcess -gt 0) {
            Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }

    Get-CimInstance Win32_Process -Filter "name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*uvicorn*app.main:app*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Test-PythonModule {
    param(
        [string]$PythonExe,
        [string]$ModuleName
    )

    try {
        $code = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)"
        & $PythonExe -c $code *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Test-Backend {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/health" -TimeoutSec 3
        return $health.status -eq "ok"
    } catch {
        return $false
    }
}

function Require-Path {
    param([string]$Path, [string]$Label)

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing $Label`: $Path"
    }
}

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$repo = $repo.Path
$envFile = Join-Path $repo ".env"
$dockerEnvFile = Join-Path $repo ".env.docker"
$appEnv = Read-DotEnv $envFile
$dockerEnv = Read-DotEnv $dockerEnvFile

$scaleDbName = "biometric_vggface2_2m_real"
$postgresUser = [string]$dockerEnv["POSTGRES_USER"]
$postgresPassword = [string]$dockerEnv["POSTGRES_PASSWORD"]
if (-not $postgresUser -or -not $postgresPassword) {
    throw "POSTGRES_USER/POSTGRES_PASSWORD are missing in .env.docker"
}

$databaseUrl = "postgresql+psycopg2://$postgresUser`:$postgresPassword@127.0.0.1:5432/$scaleDbName"
$indexRoot = "D:\datasets\vggface2_hf\benchmarks\real_image_db_index_vggface2_2m"
$customIndexPath = Join-Path $indexRoot "custom_ivfpq.faiss"
$pretrainedIndexPath = Join-Path $indexRoot "pretrained_demo.faiss"
$torchModelPath = Join-Path $repo "custom_torch_candidate_bundle\model.pth"
$onnxDetectorPath = Join-Path $repo "models\det_10g.onnx"
$onnxEmbedderPath = Join-Path $repo "models\w600k_r50.onnx"
$backendPython = Join-Path $repo "backend\.venv\Scripts\python.exe"
$desktopPython = Join-Path $repo "desktop\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $backendPython) -or -not (Test-PythonModule -PythonExe $backendPython -ModuleName "torch")) {
    $backendPython = "python"
}
if (-not (Test-PythonModule -PythonExe $backendPython -ModuleName "torch")) {
    throw "Could not find a backend Python environment with torch installed."
}
if (-not (Test-Path -LiteralPath $desktopPython)) {
    $desktopPython = "python"
}

Wait-Docker

Write-Step "Starting Docker Postgres service"
Push-Location $repo
try {
    docker compose --env-file .env.docker up -d db
} finally {
    Pop-Location
}

Write-Step "Waiting for scale database"
for ($i = 1; $i -le 60; $i++) {
    try {
        docker compose --env-file $dockerEnvFile exec -T db pg_isready -U $postgresUser -d $scaleDbName *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Postgres is ready: $scaleDbName" -ForegroundColor Green
            break
        }
    } catch {
        # Retry below.
    }
    if ($i -eq 60) {
        throw "Postgres did not become ready for database $scaleDbName."
    }
    Start-Sleep -Seconds 2
}

Write-Step "Checking runtime artifacts"
Require-Path $torchModelPath "custom Torch candidate checkpoint"
Require-Path $onnxDetectorPath "ONNX detector"
Require-Path $onnxEmbedderPath "ONNX embedder"
Require-Path $customIndexPath "custom scale index"
Require-Path $indexRoot "index directory"
Write-Host "Runtime artifacts are present." -ForegroundColor Green

if ($CheckOnly) {
    Write-Step "CheckOnly finished"
    exit 0
}

New-Item -ItemType Directory -Force -Path (Join-Path $repo "tmp") | Out-Null
$backendRuntimeScript = Join-Path $repo "tmp\start_scale_backend_runtime.ps1"
$desktopRuntimeScript = Join-Path $repo "tmp\start_scale_desktop_runtime.ps1"

Write-Step "Starting backend window"
Stop-BackendOnPort -Port 8000

$backendLines = [System.Collections.Generic.List[string]]::new()
$backendLines.Add('$Host.UI.RawUI.WindowTitle = "DiplomaWork Backend - Scale Full"')
$backendLines.Add('$ErrorActionPreference = "Stop"')
Add-EnvLines -Values $appEnv -Lines $backendLines
$backendLines.Add(('$env:DATABASE_URL = ' + (Quote-Ps $databaseUrl)))
$backendLines.Add('$env:DEFAULT_PIPELINE = "custom"')
$backendLines.Add('$env:ENABLE_PRETRAINED_PIPELINE = "true"')
$backendLines.Add('$env:ENABLE_CUSTOM_PIPELINE = "true"')
$backendLines.Add('$env:PRETRAINED_BACKEND = "onnx"')
$backendLines.Add('$env:CUSTOM_BACKEND = "torch"')
$backendLines.Add(('$env:ONNX_DETECTOR_PATH = ' + (Quote-Ps $onnxDetectorPath)))
$backendLines.Add(('$env:ONNX_EMBEDDER_PATH = ' + (Quote-Ps $onnxEmbedderPath)))
$backendLines.Add(('$env:TORCH_MODEL_PATH = ' + (Quote-Ps $torchModelPath)))
$backendLines.Add('$env:TORCH_MODEL_ARCH = "insightface_iresnet100"')
$backendLines.Add('$env:TORCH_PREPROCESS = "runtime_fallback_center_crop"')
$backendLines.Add('$env:TORCH_TTA = "hflip"')
$backendLines.Add('$env:TORCH_NORM_EMBEDDINGS = "true"')
$backendLines.Add('$env:TORCH_USE_FP16 = "true"')
$backendLines.Add('$env:TORCH_DEVICE = "cuda"')
$backendLines.Add('$env:CUSTOM_DETECTION_BACKEND = "opencv"')
$backendLines.Add('$env:CUSTOM_ALLOW_CENTER_CROP = "true"')
$backendLines.Add('$env:CUSTOM_MATCH_THRESHOLD = "0.205047"')
$backendLines.Add('$env:CUSTOM_LIVE_MATCH_THRESHOLD = "0.205047"')
$backendLines.Add('$env:PRETRAINED_MATCH_THRESHOLD = "0.40"')
$backendLines.Add(('$env:CUSTOM_INDEX_PATH = ' + (Quote-Ps $customIndexPath)))
$backendLines.Add(('$env:PRETRAINED_INDEX_PATH = ' + (Quote-Ps $pretrainedIndexPath)))
$backendLines.Add('$env:INDEX_TYPE = "ivfpq"')
$backendLines.Add('$env:IVFPQ_NLIST = "4096"')
$backendLines.Add('$env:IVFPQ_M = "32"')
$backendLines.Add('$env:IVFPQ_NBITS = "8"')
$backendLines.Add('$env:IVFPQ_NPROBE = "32"')
$backendLines.Add('$env:IVFPQ_NPROBE_FAST = "32"')
$backendLines.Add('$env:IVFPQ_NPROBE_SAFE = "128"')
$backendLines.Add('$env:SEARCH_CANDIDATE_K = "200"')
$backendLines.Add('$env:SEARCH_CANDIDATE_K_FAST = "100"')
$backendLines.Add('$env:SEARCH_CANDIDATE_K_SAFE = "500"')
$backendLines.Add('$env:RATE_LIMIT_ENABLED = "false"')
$backendLines.Add('$env:AUTO_SAVE_INDEX = "false"')
$backendLines.Add(('$env:PYTHONPATH = ' + (Quote-Ps (Join-Path $repo "backend"))))
$backendLines.Add(('Set-Location ' + (Quote-Ps (Join-Path $repo "backend"))))
$backendLines.Add('Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue')
$backendLines.Add(('& ' + (Quote-Ps $backendPython) + ' -m uvicorn app.main:app --host 127.0.0.1 --port 8000'))
Set-Content -LiteralPath $backendRuntimeScript -Value $backendLines -Encoding UTF8

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $backendRuntimeScript
)

for ($i = 1; $i -le 90; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Backend) {
        Write-Host "Backend is ready: http://127.0.0.1:8000" -ForegroundColor Green
        break
    }
    if ($i -eq 90) {
        throw "Backend did not answer within 90 seconds. Check the backend window."
    }
}

Write-Step "Scale counts"
$headers = @{}
if ($appEnv["API_KEY"]) {
    $headers["X-API-Key"] = $appEnv["API_KEY"]
}
$dbStats = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/database/stats" -Headers $headers -TimeoutSec 20
$customStats = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/index/stats?pipeline=custom" -Headers $headers -TimeoutSec 20
$pretrainedStats = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/index/stats?pipeline=pretrained" -Headers $headers -TimeoutSec 20
Write-Host ("Active identities: {0}" -f $dbStats.active_persons)
Write-Host ("Active templates: {0}" -f $dbStats.active_embeddings)
Write-Host ("Custom indexed vectors: {0}" -f $customStats.embeddings_count)
Write-Host ("Pretrained indexed vectors: {0}" -f $pretrainedStats.embeddings_count)

Write-Step "Starting desktop window"
$desktopLines = [System.Collections.Generic.List[string]]::new()
$desktopLines.Add('$Host.UI.RawUI.WindowTitle = "DiplomaWork Desktop - Scale Full"')
$desktopLines.Add('$ErrorActionPreference = "Stop"')
$desktopLines.Add(('$env:API_BASE_URL = ' + (Quote-Ps "http://127.0.0.1:8000")))
$desktopLines.Add(('$env:API_KEY = ' + (Quote-Ps ([string]$appEnv["API_KEY"]))))
$desktopLines.Add(('$env:ADMIN_API_KEY = ' + (Quote-Ps ([string]$appEnv["ADMIN_API_KEY"]))))
$desktopLines.Add('$env:API_TIMEOUT_SEC = "45"')
$desktopLines.Add('$env:CAMERA_FRAME_WIDTH = "1280"')
$desktopLines.Add('$env:CAMERA_FRAME_HEIGHT = "720"')
$desktopLines.Add('$env:LIVE_SCAN_INTERVAL_MS = "150"')
$desktopLines.Add('$env:LIVE_MAX_WIDTH = "640"')
$desktopLines.Add('$env:LIVE_JPEG_QUALITY = "72"')
$desktopLines.Add('$env:CUSTOM_LIVE_MAX_WIDTH = "960"')
$desktopLines.Add('$env:CUSTOM_LIVE_JPEG_QUALITY = "82"')
$desktopLines.Add(('$env:PYTHONPATH = ' + (Quote-Ps (Join-Path $repo "desktop"))))
$desktopLines.Add(('Set-Location ' + (Quote-Ps (Join-Path $repo "desktop"))))
$desktopLines.Add('Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue')
$desktopLines.Add(('& ' + (Quote-Ps $desktopPython) + ' -m app.main'))
Set-Content -LiteralPath $desktopRuntimeScript -Value $desktopLines -Encoding UTF8

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $desktopRuntimeScript
)

Write-Step "Ready"
Write-Host "Docker Postgres, local backend, and Desktop were started."
Write-Host "Backend URL: http://127.0.0.1:8000"
Write-Host "Dataset labels in the scale gallery use 'Gallery Person 000001' style."
Write-Host "Manual labels are kept as real profile labels from data\new_custom_enroll."
