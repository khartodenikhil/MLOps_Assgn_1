# Rebuild everything from scratch in a brand-new virtual environment and
# check the results match the committed metrics.
#
# Usage (from the project root, in PowerShell):
#     powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1

$ErrorActionPreference = "Stop"
$VenvDir = ".venv-repro"
$Py = Join-Path $VenvDir "Scripts\python.exe"

function Step($Message) { Write-Host "`n=== $Message ===" -ForegroundColor Cyan }

function Run {
    & $Py @args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: python $args" }
}

Step "Python version"
python --version

Step "Creating a clean virtual environment ($VenvDir)"
if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir }
python -m venv $VenvDir
if ($LASTEXITCODE -ne 0) { throw "Could not create virtual environment" }

Step "Installing pinned dependencies"
Run -m pip install --quiet --upgrade pip
Run -m pip install --quiet -r requirements.txt

Step "Saving current metrics for comparison"
$HaveBaseline = Test-Path "reports\metrics.json"
if ($HaveBaseline) { Copy-Item "reports\metrics.json" "reports\metrics_before.json" -Force }

Step "Downloading and cleaning data"
Run -m src.data --force

Step "Training (no registry update)"
Run -m src.train --no-register

Step "Predicting on sample patients"
Run -m src.predict --input samples\patients.json

if ($HaveBaseline) {
    Step "Comparing with previous metrics"
    Run scripts\compare_metrics.py reports\metrics_before.json reports\metrics.json
    Remove-Item "reports\metrics_before.json"
}

Step "Done. Remove the test environment with: Remove-Item -Recurse -Force $VenvDir"
