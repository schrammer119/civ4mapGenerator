$activationScript = Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"

if (-not (Test-Path $activationScript)) {
    throw "The virtual environment does not exist. Run .\scripts\setup-venv.ps1 first."
}

. $activationScript
