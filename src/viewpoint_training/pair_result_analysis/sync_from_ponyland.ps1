param(
    [string]$RemoteHost = "lisannehuisman@wildfire.science.ru.nl",
    [string]$RemoteExperimentRoot = "/vol/tensusers6/lisannehuisman/experiments/m4_yolov8l_pair_training",
    [string]$LocalSnapshotRoot = ".\m4_pair_partial_analysis\data\current_snapshot"
)

$ErrorActionPreference = "Stop"

$snapshotRoot = Resolve-Path "." | ForEach-Object {
    Join-Path $_.Path $LocalSnapshotRoot.TrimStart(".\")
}

$reportsDir = Join-Path $snapshotRoot "reports"
$plotsDir = Join-Path $snapshotRoot "plots"
$manifestsDir = Join-Path $snapshotRoot "manifests"

New-Item -ItemType Directory -Force $reportsDir | Out-Null
New-Item -ItemType Directory -Force $plotsDir | Out-Null
New-Item -ItemType Directory -Force $manifestsDir | Out-Null

$filesToCopy = @(
    @{ Remote = "$RemoteExperimentRoot/reports/master_results.csv"; Local = $reportsDir; Optional = $false }
    @{ Remote = "$RemoteExperimentRoot/reports/master_results_option_a.csv"; Local = $reportsDir; Optional = $false }
    @{ Remote = "$RemoteExperimentRoot/reports/master_results_option_b.csv"; Local = $reportsDir; Optional = $false }
    @{ Remote = "$RemoteExperimentRoot/reports/pair_subset_experiment_report.md"; Local = $reportsDir; Optional = $false }
    @{ Remote = "$RemoteExperimentRoot/plots/top_performing_pairs.png"; Local = $plotsDir; Optional = $false }
    @{ Remote = "$RemoteExperimentRoot/plots/pair_performance_distribution.png"; Local = $plotsDir; Optional = $false }
    @{ Remote = "$RemoteExperimentRoot/plots/pair_performance_heatmap.png"; Local = $plotsDir; Optional = $false }
    @{ Remote = "$RemoteExperimentRoot/plots/gain_loss_vs_full_m4.png"; Local = $plotsDir; Optional = $true }
    @{ Remote = "$RemoteExperimentRoot/manifests/viewpoint_pairs.csv"; Local = $manifestsDir; Optional = $false }
    @{ Remote = "$RemoteExperimentRoot/manifests/viewpoint_inventory.csv"; Local = $manifestsDir; Optional = $true }
    @{ Remote = "$RemoteExperimentRoot/manifests/subset_build_metadata.json"; Local = $manifestsDir; Optional = $true }
    @{ Remote = "$RemoteExperimentRoot/last_full_jobid.txt"; Local = $snapshotRoot; Optional = $true }
)

$syncedFiles = @()
foreach ($item in $filesToCopy) {
    & scp "$RemoteHost`:$($item.Remote)" $item.Local
    if ($LASTEXITCODE -eq 0) {
        $syncedFiles += $item.Remote
        continue
    }

    if ($item.Optional) {
        Write-Warning "Skipping optional remote file that was not copied: $($item.Remote)"
        continue
    }

    throw "Failed to copy required remote file: $($item.Remote)"
}

$syncedAt = (Get-Date).ToString("o")
$manifest = [ordered]@{
    remote_host = $RemoteHost
    remote_experiment_root = $RemoteExperimentRoot
    local_snapshot_root = $snapshotRoot
    synced_at = $syncedAt
    files = $syncedFiles
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $snapshotRoot "sync_manifest.json")

Write-Host "Synced Ponyland partial results into $snapshotRoot"
