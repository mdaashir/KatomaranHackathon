# Models

## `yolov8n-face.pt`

- **Purpose:** YOLOv8 nano face detector used by `src/detection`.
- **Source:** https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt
- **Size:** 6,247,374 bytes
- **SHA256:** `6217b2550b792b789b3e42a55453c6c4326d6293d27f1a6ae56da715c9fbc668`
- **Framework:** Ultralytics YOLOv8, loaded with `ultralytics.YOLO`.
- **Extra dependency:** this checkpoint references `omegaconf`; it is pinned in
  `requirements.txt`. Ultralytics does not install it automatically.
- **License:** Ultralytics YOLOv8 is AGPL-3.0. Verify the license before any
  commercial use.
- **Security:** the `.pt` file is a Python pickle. Only load weights from a
  source you trust; never load a checkpoint supplied by an untrusted third
  party.
- **Not committed:** all model weights are excluded from git. Fetch and verify
  this checkpoint with `scripts/download_models.py`.

**Checkpoint behaviour:** this checkpoint is **detect-only** (`task=detect`,
one class `face`). It does not emit the five ArcFace keypoints. The detector
therefore returns `FaceDetection.keypoints = None`; Phase 4 derives alignment
points with the InsightFace landmark model instead.

To download and verify every model:

```powershell
uv run --no-sync python scripts\download_models.py
```
