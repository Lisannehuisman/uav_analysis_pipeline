$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $PSScriptRoot
Set-Location $workspace

$env:YOLO_CONFIG_DIR = (Resolve-Path ".\Ultralytics").Path

& ".\.venv\Scripts\python.exe" ".\probability_fusion\run_full72_pipeline.py" --overwrite
