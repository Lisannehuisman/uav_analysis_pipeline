# Training duration summary

- YOLOv8n: mean duration 1.97h across 5 regimes.
- YOLOv8l: mean duration 2.36h across 5 regimes.
- Faster R-CNN: mean duration 0.08h across 5 regimes.

Method notes:
- YOLO durations are read directly from the final cumulative `time` column in `results.csv`.
- Faster R-CNN durations are estimated from logged iteration time in `metrics.json` because an explicit cumulative wall-clock total was not stored.