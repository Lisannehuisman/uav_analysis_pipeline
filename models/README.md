# Model Checkpoints

Model checkpoints are not included in the standalone thesis archive because they are large and can be regenerated from the training scripts.

Scripts expect restored checkpoints under paths such as:

```text
models/yolov8l/S0_M4_yolov8l/weights/best.pt
models/yolov8n/S0_M4_yolov8n/weights/best.pt
models/faster_rcnn/S0_M4_run1/
models/real_uav_finetuned/weights/last.pt
```

The result tables and figures in `results/` preserve the thesis measurements without requiring these checkpoints to be present.
