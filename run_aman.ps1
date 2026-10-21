param(
    [ValidateSet("account-only", "dry-run", "submit-approved")]
    [string]$Mode = "dry-run"
)
$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "portfolio_runner.py"
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    & $VenvPython $Runner --owner aman --mode $Mode
} else {
    python $Runner --owner aman --mode $Mode
}
exit $LASTEXITCODE
