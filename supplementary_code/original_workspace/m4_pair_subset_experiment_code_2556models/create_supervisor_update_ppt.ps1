param(
    [string]$DataJson = "outputs\m4_pair_subset_experiment\presentation\supervisor_update_data.json",
    [string]$OutputPath = "outputs\m4_pair_subset_experiment\presentation\supervisor_update_multiviewpoint_training.pptx"
)

$ErrorActionPreference = "Stop"

function Add-TitleSlide {
    param(
        $Presentation,
        [string]$Title,
        [string]$Subtitle
    )

    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, 12)
    $titleShape = $slide.Shapes.AddTextbox(1, 40, 40, 860, 60)
    $titleShape.TextFrame.TextRange.Text = $Title
    $titleShape.TextFrame.TextRange.Font.Size = 28
    $titleShape.TextFrame.TextRange.Font.Bold = $true

    $subtitleShape = $slide.Shapes.AddTextbox(1, 40, 120, 860, 120)
    $subtitleShape.TextFrame.TextRange.Text = $Subtitle
    $subtitleShape.TextFrame.TextRange.Font.Size = 18
    $subtitleShape.TextFrame.TextRange.ParagraphFormat.Bullet.Visible = 0
    return $slide
}

function Add-BulletSlide {
    param(
        $Presentation,
        [string]$Title,
        [string[]]$Bullets,
        [string]$ImagePath = ""
    )

    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, 12)
    $titleShape = $slide.Shapes.AddTextbox(1, 40, 25, 860, 45)
    $titleShape.TextFrame.TextRange.Text = $Title
    $titleShape.TextFrame.TextRange.Font.Size = 24
    $titleShape.TextFrame.TextRange.Font.Bold = $true

    $textWidth = if ([string]::IsNullOrWhiteSpace($ImagePath)) { 860 } else { 430 }
    $bulletShape = $slide.Shapes.AddTextbox(1, 40, 90, $textWidth, 420)
    $textRange = $bulletShape.TextFrame.TextRange
    $textRange.Text = ($Bullets -join "`r")
    $textRange.Font.Size = 18
    $bulletShape.TextFrame.WordWrap = -1
    $bulletShape.TextFrame.AutoSize = 1

    for ($i = 1; $i -le $Bullets.Count; $i++) {
        $paragraph = $textRange.Paragraphs($i)
        $paragraph.ParagraphFormat.Bullet.Visible = -1
        $paragraph.ParagraphFormat.Bullet.Type = 1
        $paragraph.ParagraphFormat.SpaceAfter = 8
    }

    if (-not [string]::IsNullOrWhiteSpace($ImagePath) -and (Test-Path $ImagePath)) {
        $slide.Shapes.AddPicture((Resolve-Path $ImagePath), $false, $true, 500, 110, 400, 300) | Out-Null
    }
    return $slide
}

if (-not (Test-Path $DataJson)) {
    throw "Could not find data JSON: $DataJson"
}

$data = Get-Content $DataJson -Raw | ConvertFrom-Json
$presentationDir = Split-Path -Parent (Resolve-Path $DataJson)
$outputResolved = Join-Path (Get-Location) $OutputPath
$outputDir = Split-Path -Parent $outputResolved
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$ppt = $null
try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = -1
    $presentation = $ppt.Presentations.Add()

    $subtitle = @(
        "Supervisor update"
        $data.update_date
        "Pilot status: $($data.current_pilot_status.completed_option_a_of_5) of 5 pairs completed end-to-end"
    ) -join "`r"
    Add-TitleSlide -Presentation $presentation -Title $data.presentation_title -Subtitle $subtitle | Out-Null

    Add-BulletSlide -Presentation $presentation -Title "Research Question" -Bullets @(
        "Goal: quantify how much detector generalization can be learned from training on only two M4 viewpoints."
        "Secondary goal: estimate which viewpoints are most valuable by aggregating pair-trained model performance."
        "This is a training-subset experiment, not multiview inference or fusion."
        "Primary evaluation uses Option A: test each pair-trained model on the full fixed M4 test set across all 72 viewpoints."
        "Option B is still computed as a diagnostic but is not the headline result."
        "Here, viewpoint contribution means a simple pair-aggregation score over viewpoint subsets, not late-fusion Shapley."
    ) | Out-Null

    Add-BulletSlide -Presentation $presentation -Title "Training Protocol" -Bullets @(
        "For each viewpoint pair {i, j}, we build a subset using all M4 train images from viewpoints i and j."
        "Labels stay unchanged and the original train, val, and test split boundaries are preserved."
        "Each pair gets its own standard YOLOv8l model with the same pretrained initialization and same settings."
        "Because only the viewpoint composition changes, pair scores are directly comparable."
    ) -ImagePath (Join-Path $presentationDir "pilot_pair_overview.png") | Out-Null

    Add-BulletSlide -Presentation $presentation -Title "Viewpoint Contribution Scoring" -Bullets @(
        "We define a value function v(S) as the full-test detection score of a model trained only on viewpoint set S."
        "In the current sweep, S is a pair and the main value is Option A mAP50-95."
        "For viewpoint i, we estimate its contribution by averaging the values of all trained pairs that contain i."
        "A viewpoint gets a high score if it repeatedly participates in pair-trained models that generalize well."
    ) | Out-Null

    Add-BulletSlide -Presentation $presentation -Title "How We Will Find the Best Angles" -Bullets @(
        "Rank all 2556 viewpoint pairs by Option A mAP50-95, with mAP50 and F1 as supporting metrics."
        "Build a 72 x 72 pair-performance heatmap to identify strong and weak viewpoint combinations."
        "Aggregate pair scores back to each viewpoint with the viewpoint contribution score."
        "Then compare patterns across azimuth, elevation, and radius to see which angle families help most."
    ) -ImagePath (Join-Path $presentationDir "pilot_progress.png") | Out-Null

    Add-BulletSlide -Presentation $presentation -Title "Ponyland Pilot Rollout" -Bullets @(
        "Launched 5 representative pilot pairs to validate the end-to-end workflow before the 2556-pair sweep."
        "Initial run with batch 16 failed with CUDA out-of-memory on the available Ponyland GPUs."
        "Rerun with batch 4 and eval batch 8 is stable and training progresses normally."
        "All 5 pilot pairs have now completed successfully, so the pipeline is working on-cluster end to end."
    ) | Out-Null

    $best = $data.best_so_far
    Add-BulletSlide -Presentation $presentation -Title "Pilot Results" -Bullets @(
        "Current pilot status: $($data.current_pilot_status.completed_option_a_of_5) of 5 pilot pairs completed on Option A."
        "Best pair so far: $($best.pair_id) = $($best.label)"
        ("Best-so-far full-test metrics: F1={0:N3}, mAP50={1:N3}, mAP50-95={2:N3}" -f $best.metrics.f1, $best.metrics.map50, $best.metrics.map50_95)
        ("Full M4 baseline: F1={0:N3}, mAP50={1:N3}, mAP50-95={2:N3}" -f $data.baseline_metrics.f1, $data.baseline_metrics.map50, $data.baseline_metrics.map50_95)
    ) -ImagePath (Join-Path $presentationDir "best_so_far_vs_baseline.png") | Out-Null

    Add-BulletSlide -Presentation $presentation -Title "Interpretation So Far" -Bullets @(
        "Training on only two viewpoints can already produce meaningful generalization, but current best pilot performance is still below the full M4 baseline."
        "Pair geometry appears to matter: the current leader varies elevation while holding azimuth and radius fixed."
        "The pilot is now complete, so the next decision is whether the full 2556-pair sweep is worth the compute cost."
    ) | Out-Null

    Add-BulletSlide -Presentation $presentation -Title "Next Steps" -Bullets @(
        "Use the full pilot set to decide whether the pair-sweep is scientifically informative enough to justify launching all 2556 pairs."
        "If we launch the full sweep, keep the Ponyland-safe batch setting fixed across every pair run."
        "For the cleanest comparison, rerun the full M4 YOLOv8l baseline once with the same Ponyland training settings."
        "Prepare a short supervisor decision memo summarizing pilot findings, compute cost, and expected scientific payoff."
    ) | Out-Null

    $presentation.SaveAs($outputResolved)
    $presentation.Close()
    $ppt.Quit()
    Write-Output "Saved PowerPoint: $outputResolved"
}
catch {
    if ($null -ne $presentation) {
        $presentation.Close()
    }
    if ($null -ne $ppt) {
        $ppt.Quit()
    }
    throw
}
