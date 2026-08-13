#Requires -Version 5.1
<#
.SYNOPSIS
    Image Compare -- launcher for PowerShell.

.DESCRIPTION
    Creates/reuses a local .venv, then runs pip install on every launch
    (fast and idempotent when nothing changed -- typically 1-3 seconds),
    optionally installs the CLIP/DINOv2 extras, then starts the server.

.PARAMETER WithClip
    Also install the optional CLIP dependencies (torch, open_clip_torch).
    Heavy download; only needed for "CLIP (semantic)" mode.

.PARAMETER WithDino
    Also install the optional DINOv2 dependencies (torch, transformers).
    Heavy download (though smaller than CLIP's); only needed for
    "DINOv2 (visual)" mode -- the recommended mode for visual similarity.

.PARAMETER Port
    Port to serve on. Default: 8000.

.PARAMETER Reload
    Run uvicorn with --reload (auto-restart on code changes). Useful for
    development.

.PARAMETER ForceReinstall
    Force pip to reinstall all packages from scratch (--force-reinstall)
    instead of the normal fast idempotent check.

.EXAMPLE
    ./run.ps1
.EXAMPLE
    ./run.ps1 -WithDino -Port 8080
#>

[CmdletBinding()]
param(
    [switch]$WithClip,
    [switch]$WithDino,
    [int]$Port = 8000,
    [switch]$Reload,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "[SETUP] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Start($msg){ Write-Host "[START] $msg" -ForegroundColor Yellow }

$ProjectDir  = $PSScriptRoot
$VenvDir     = Join-Path $ProjectDir ".venv"

Write-Host ""
Write-Host "=== Image Compare ===" -ForegroundColor Magenta
Write-Host ""

Push-Location $ProjectDir

try {

# ---- 1. find a *working* Python interpreter (prefer the py launcher) ----
# Get-Command only proves a name exists on PATH, not that it actually runs
# (PATH can contain stale/broken entries, e.g. a partially uninstalled
# version) -- so each candidate is test-invoked and only kept if it
# genuinely responds.
function Get-PythonCommand {
    $candidates = @(
        @{ Cmd = "py"; Args = @("-3") },
        @{ Cmd = "py"; Args = @() },
        @{ Cmd = "python3"; Args = @() },
        @{ Cmd = "python"; Args = @() }
    )
    foreach ($c in $candidates) {
        $found = Get-Command $c.Cmd -ErrorAction SilentlyContinue
        if (-not $found) { continue }
        try {
            & $c.Cmd @($c.Args + "--version") 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return $c
            }
        } catch {
            continue
        }
    }
    return $null
}

$pyCmd = Get-PythonCommand
if (-not $pyCmd) {
    Write-Err "No working Python 3.10+ interpreter was found on PATH."
    Write-Host "        Tried: py -3, py, python3, python"
    Write-Host ""
    Write-Host "        If Python is not installed, get it from:"
    Write-Host "            https://www.python.org/downloads/"
    Write-Host "        and check 'Add python.exe to PATH' during setup."
    Write-Host ""
    Write-Host "        If Python IS installed but this still fails, PATH may have"
    Write-Host "        a stale entry pointing at a python.exe that no longer exists."
    Write-Host "        Try opening a brand-new terminal window, or check what PATH"
    Write-Host "        currently resolves to with:"
    Write-Host "            where.exe python"
    exit 1
}

$versionOutput = & $pyCmd.Cmd @($pyCmd.Args + "--version") 2>&1
if ($versionOutput -notmatch "Python (\d+)\.(\d+)") {
    Write-Err "Could not determine Python version from: $versionOutput"
    exit 1
}
$major = [int]$Matches[1]
$minor = [int]$Matches[2]
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
    Write-Err "Python $major.$minor found, but 3.10+ is required."
    exit 1
}
Write-Ok "Using Python $major.$minor"

# ---- 2. create venv if missing ----
$venvPython = if ($IsWindows -or $env:OS -match "Windows") {
    Join-Path $VenvDir "Scripts\python.exe"
} else {
    Join-Path $VenvDir "bin/python"
}

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment in .venv ..."
    & $pyCmd.Cmd @($pyCmd.Args + @("-m", "venv", $VenvDir))
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Ok "Virtual environment already exists"
}

# ---- 3. install / verify dependencies ----
# pip itself is fast and idempotent when nothing has changed (it just
# verifies installed versions already satisfy the spec, typically 1-3
# seconds) -- so this always runs pip rather than trying to detect "did
# dependencies change" ourselves. A prior version compared file
# timestamps and skipped this step when they looked up to date, but
# timestamp comparison is fragile (e.g. zip extraction can leave
# pyproject.toml looking "older" than a previously created marker file)
# and went stale the moment a new dependency was added (e.g. openpyxl),
# silently leaving it uninstalled.
Write-Step "Checking dependencies ..."
& $venvPython -m pip install --upgrade pip --quiet
if ($ForceReinstall) {
    & $venvPython -m pip install --force-reinstall -e ".[dev]"
} else {
    & $venvPython -m pip install -e ".[dev]" --quiet
}
if ($LASTEXITCODE -ne 0) {
    Write-Err "Dependency installation failed. See output above."
    exit 1
}
Write-Ok "Dependencies ready"

# ---- 4. optional CLIP / DINOv2 extras ----
if ($WithClip) {
    & $venvPython -c "import open_clip, torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Step "Installing optional CLIP dependencies (large download) ..."
        & $venvPython -m pip install -e ".[clip]"
        if ($LASTEXITCODE -ne 0) {
            Write-Err "CLIP dependency installation failed."
            exit 1
        }
    } else {
        Write-Ok "CLIP dependencies already installed"
    }
}
if ($WithDino) {
    & $venvPython -c "import transformers, torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Step "Installing optional DINOv2 dependencies (large download) ..."
        & $venvPython -m pip install -e ".[dino]"
        if ($LASTEXITCODE -ne 0) {
            Write-Err "DINOv2 dependency installation failed."
            exit 1
        }
    } else {
        Write-Ok "DINOv2 dependencies already installed"
    }
}

# ---- 5. run the server ----
Write-Host ""
Write-Start "Launching Image Compare at http://127.0.0.1:$Port"
Write-Host "        Press CTRL+C to stop."
Write-Host ""

$uvicornArgs = @("-m", "uvicorn", "imagecompare.main:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Reload) {
    $uvicornArgs += "--reload"
}

& $venvPython @uvicornArgs

}
finally {
    Pop-Location
}
