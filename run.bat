@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "VENV_DIR=%PROJECT_DIR%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

set "WITH_CLIP=0"
set "WITH_DINO=0"
set "FORCE=0"
set "ARGS=%*"
echo %ARGS%| find "--with-clip" >nul
if not errorlevel 1 set "WITH_CLIP=1"
echo %ARGS%| find "--with-dino" >nul
if not errorlevel 1 set "WITH_DINO=1"
echo %ARGS%| find "--force" >nul
if not errorlevel 1 set "FORCE=1"

echo.
echo === Image Compare (launcher v3) ===
echo.

REM ---- 1. find a *working* Python interpreter ----
REM      "where" only proves a name exists on PATH, not that it actually
REM      runs (PATH can contain stale/broken entries, e.g. a partially
REM      uninstalled version) - so each candidate is test-invoked and
REM      only kept if it genuinely responds.
set "PY="
for %%C in ("py -3" "py" "python3" "python") do (
    if "!PY!"=="" (
        %%~C --version >nul 2>nul
        if not errorlevel 1 set "PY=%%~C"
    )
)

if "%PY%"=="" (
    echo [ERROR] No working Python 3.10+ interpreter was found on PATH.
    echo         Tried: py -3, py, python3, python
    echo.
    echo         If Python is not installed, get it from:
    echo             https://www.python.org/downloads/
    echo         and check "Add python.exe to PATH" during setup.
    echo.
    echo         If Python IS installed but this still fails, PATH may have
    echo         a stale entry pointing at a python.exe that no longer
    echo         exists. Try opening a brand-new terminal window, or check
    echo         what PATH currently resolves to with:
    echo             where python
    pause
    exit /b 1
)

REM ---- 2. verify Python version is 3.10+ (let Python itself judge, not batch string math) ----
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.10+ is required. Found:
    %PY% --version
    pause
    exit /b 1
)
for /f "delims=" %%v in ('%PY% --version') do echo [OK] Using %%v

REM ---- 3. create venv if missing ----
if exist "%VENV_PY%" (
    echo [OK] Virtual environment already exists
) else (
    echo [SETUP] Creating virtual environment in .venv ...
    %PY% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM ---- 4. install / verify dependencies ----
REM      pip itself is fast and idempotent when nothing has changed (it
REM      just verifies installed versions already satisfy the spec,
REM      typically 1-3 seconds) -- so this always runs pip rather than
REM      trying to detect "did dependencies change" ourselves. A prior
REM      version skipped this step whenever a marker file existed, which
REM      went stale the moment a new dependency was added to
REM      pyproject.toml (e.g. openpyxl) and silently left it uninstalled.
echo [SETUP] Checking dependencies ...
"%VENV_PY%" -m pip install --upgrade pip --quiet
if "%FORCE%"=="1" (
    "%VENV_PY%" -m pip install --force-reinstall -e ".[dev]"
) else (
    "%VENV_PY%" -m pip install -e ".[dev]" --quiet
)
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See output above.
    pause
    exit /b 1
)
echo [OK] Dependencies ready

REM ---- 5. optional CLIP / DINOv2 extras ----
if "%WITH_CLIP%"=="1" (
    "%VENV_PY%" -c "import open_clip, torch" >nul 2>nul
    if errorlevel 1 (
        echo [SETUP] Installing optional CLIP dependencies ^(large download^) ...
        "%VENV_PY%" -m pip install -e ".[clip]"
    ) else (
        echo [OK] CLIP dependencies already installed
    )
)
if "%WITH_DINO%"=="1" (
    "%VENV_PY%" -c "import transformers, torch" >nul 2>nul
    if errorlevel 1 (
        echo [SETUP] Installing optional DINOv2 dependencies ^(large download^) ...
        "%VENV_PY%" -m pip install -e ".[dino]"
    ) else (
        echo [OK] DINOv2 dependencies already installed
    )
)

REM ---- 6. run the server ----
echo.
echo [START] Launching Image Compare at http://127.0.0.1:8000
echo         Press CTRL+C to stop.
echo.
"%VENV_PY%" -m uvicorn imagecompare.main:app --host 127.0.0.1 --port 8000

endlocal
