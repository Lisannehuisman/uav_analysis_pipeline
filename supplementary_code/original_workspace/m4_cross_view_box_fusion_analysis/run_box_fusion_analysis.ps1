param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

$pythonCandidates = @(
    (Join-Path $projectRoot ".venv\Scripts\python.exe"),
    "python"
)

$pythonExe = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $pythonExe) {
    throw "Could not find a Python interpreter. Expected one of: $($pythonCandidates -join ', ')"
}

$scriptPath = Join-Path $scriptDir "run_box_fusion_analysis.py"
Write-Host "Using Python: $pythonExe"
& $pythonExe $scriptPath @ScriptArgs

