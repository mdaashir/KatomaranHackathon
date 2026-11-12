# InsightFace models

The tracker uses two ONNX models from the official InsightFace `buffalo_l`
pack:

| File | Size | SHA256 |
|---|---:|---|
| `w600k_r50.onnx` | 174,383,860 B | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| `2d106det.onnx` | 5,030,888 B | `f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf` |

- **Source:** https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
- **Purpose:** `w600k_r50.onnx` is the ArcFace recognizer (512-d);
  `2d106det.onnx` supplies the 106 landmarks used to derive the 5 ArcFace
  alignment points.
- **License:** InsightFace models are released for non-commercial research.
  Check the InsightFace license before any commercial use.

The files are **not committed** because the recognizer is 174 MB, above
GitHub's per-file limit. Download and verify them with:

```powershell
uv run --no-sync python scripts\download_models.py
```

The script extracts only these two files, verifies the pinned checksums, and
removes the 275 MB archive unless `--keep-zip` is passed.

For CUDA execution, run `scripts/setup_gpu.ps1` first. ONNX Runtime GPU 1.22
targets CUDA 12.x; it uses the CUDA/cuDNN DLLs bundled with the CUDA PyTorch
build.
