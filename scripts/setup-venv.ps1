$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirements = Join-Path $repoRoot "requirements.txt"

if (-not (Test-Path $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $venvPath
    } else {
        & python -m venv $venvPath
    }
}

if (-not (Test-Path $venvPython)) {
    throw "Could not create the virtual environment at $venvPath"
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --upgrade -r $requirements
Write-Host "Development environment ready: $venvPython"
