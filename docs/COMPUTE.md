# COMPUTE — CPU, GPU, Memory, Storage

> All numbers are **engineering estimates or measured values** for the chosen
> stack. Rows marked **measured** come from the target machine; the remaining
> ranges are planning context. Phase 3–6 measured the individual stages and
> Phase 8 measured the complete 22-clip pass below.
>
> Phase 8 used ad-hoc timing runs; a separate `scripts/benchmark.py` was not
> needed. The measured rows below are the current numbers.
>
> Requirement audit: [`COMPLIANCE.md`](./COMPLIANCE.md).

---

## 1. Workload

### Primary dataset (`src/video/record_*.mp4`)

| Property | Value |
|---|---|
| Files | 22 independent 60 s clips |
| Resolution | 2688 × 1520 (4.09 MP) |
| Codec / FPS | HEVC / 25 |
| Total frames | 22 × 1 502 = **33 044** |
| Total wall time | **22 min** |
| Traffic | Sparse: 0–6 people per frame, 1–4 new faces per clip |
| Face sizes | ~25–45 px far, ~60–130 px near |
| Night / IR | No; indoor reflected lighting |

### Stress dataset (`video_sample1.mp4`)

| Property | Value |
|---|---|
| Resolution / codec | 3840 × 2160 (8.3 MP), H.264 |
| FPS / frames | 29.97 / 240 (8.0 s) |
| Visible faces | 25–40 per frame, high motion |

### Processing cadence

`detection_rate = fps / frame_skip`

| Profile | FPS | `frame_skip` | Cycles/s |
|---|---:|---:|---:|
| CPU real-time (RTSP) | 25 | 4 | 6.25 |
| CPU file | 25 | 3 | 8.33 |
| GPU file / RTSP | 25 | 3 | 8.33 |
| 4K stress | 29.97 | 3 | 9.99 |

---

## 2. Reference hardware

| Tier | Assumption |
|---|---|
| **CPU** | 8 logical cores, modern x86 (e.g. i7-11800H / Ryzen 7 5800H), ONNX Runtime CPU EP, 4 intra-op threads |
| **GPU** | NVIDIA T4 16 GB or RTX 3050 6 GB, CUDA EP, ONNX Runtime GPU |
| **Disk** | SSD; `fsync` 1–5 ms per event image |
| **RAM** | 16 GB minimum |
| **Decoder** | PyAV / FFmpeg software decode, single-threaded per stream |

If the interview machine is a 4-core laptop with no GPU, use the **CPU-RTSP**
profile (§6) and expect the lower end of every range.

---

## 3. Model footprint

| Component | Model | Params | Weights on disk | Runtime RAM (approx.) |
|---|---|---:|---:|---:|
| Face detection | `yolov8n-face.pt` | 3.2 M | 6.2 MB | 250–400 MB process |
| Face recognition | InsightFace `buffalo_l` → `w600k_r50` | 43.6 M | ~166 MB ONNX | 250–350 MB |
| Alignment | keypoint warp (no model) | — | — | < 5 MB |
| Tracking | ByteTrack (no model) | — | — | < 10 MB |

Notes:

- Only the **recognition** ONNX model from `buffalo_l` is loaded; face
  detection is done by YOLO, so InsightFace’s detector is never run.
- The full `buffalo_l` pack is ~190 MB on disk (detector, landmarks, gender/age
  are unused); the loader can be restricted to the recognition model to save
  ~25 MB and startup time.
- First run downloads `buffalo_l` to `~/.insightface/models/` unless vendored.

---

## 4. Per-stage latency (estimates)

| Stage | CPU 8-core | GPU (T4) | Notes |
|---|---:|---:|---|
| HEVC decode 2688×1520 | **12–22 ms/frame** | 12–22 ms (CPU) | Software decode; ~30–55 % of one core at 25 fps |
| H.264 decode 3840×2160 | 25–35 ms/frame | 25–35 ms (CPU) | 4K stress clip |
| Resize / letterbox to `imgsz` | 1.5–3 ms | 0.5–1 ms | |
| **YOLOv8n-face @ 640** | 15–30 ms | 2–3 ms | Reference point |
| **YOLOv8n-face @ 1280** | **60–120 ms** | 5–9 ms | ~4× the 640 cost |
| **YOLOv8n-face @ 1600** | **95–170 ms** | 8–12 ms | Recommended CPU default |
| **YOLOv8n-face @ 1920** | **135–270 ms** | 10–16 ms | Recommended GPU default |
| ByteTrack update | 1–3 ms/cycle | 1–3 ms | CPU only |
| Quality gate (Laplacian + geometry) | 0.5–1.5 ms/face | 0.5–1.5 ms | CPU |
| 5-point alignment warp | 0.3–0.8 ms/face | 0.3–0.8 ms | CPU |
| **ArcFace embedding 112×112** | **7–15 ms/face** | 1.5–3 ms/face | Recognised tracks are not re-embedded |
| Cosine match, 1 000 faces | 0.05 ms/query | 0.05 ms | In-memory dot product |
| Cosine match, 100 000 faces | 3–5 ms/query | 3–5 ms | Still acceptable |
| JPEG encode 256×256 q90 | 2–4 ms/event | 2–4 ms | Events are rare |
| Atomic image write + fsync | 1–5 ms/event | 1–5 ms | SSD |
| SQLite WAL transaction | 0.5–2 ms/event | 0.5–2 ms | `synchronous=NORMAL` |

### Derived per-frame cost

`avg_frame_ms = decode + (detect + track) / frame_skip + embedding_amortised + overhead`

| Scenario | Decode | Detect/cycle | Skip | Avg/frame | Max throughput | Real-time factor |
|---|---:|---:|---:|---:|---:|---:|
| CPU sparse, `imgsz=1280`, skip 3 | 17 ms | 90 ms | 3 | **≈51 ms** | 19.6 fps | **0.78×** |
| CPU sparse, `imgsz=1280`, skip 4 | 17 ms | 90 ms | 4 | **≈42 ms** | 23.8 fps | **0.95×** |
| CPU sparse, `imgsz=1600`, skip 4 | 17 ms | 130 ms | 4 | **≈51 ms** | 19.6 fps | 0.78× |
| GPU sparse, `imgsz=1920`, skip 3 | 17 ms | 14 ms | 3 | **≈22 ms** | 45 fps | **1.8×** |
| 4K stress, CPU `imgsz=1280`, skip 3 | 30 ms | 100 ms | 3 | **≈65 ms** | 15.4 fps | 0.51× |
| 4K stress, GPU `imgsz=1280`, skip 3 | 30 ms | 10 ms | 3 | **≈34 ms** | 29 fps | **0.97×** |

Embedding cost is excluded from the steady-state column because it is
per-track, not per-frame. Add `new_faces_per_second × 10 ms` (CPU) or
`× 2 ms` (GPU) for registration-heavy periods. At 2 new faces/s that is 20 ms/s
— negligible.

---

## 5. Expected runtimes on the 22-minute dataset

| Profile | Effective FPS | 33 044 frames | Notes |
|---|---:|---:|---|
| CPU, `imgsz=1280`, skip 3 | 11–12 | **47–49 min** | CPU-only Torch build |
| CPU, `imgsz=1280`, skip 4 | 14–15 | **39–41 min** | Best CPU fallback |
| CPU, `imgsz=1600`, skip 4 | 10–11 | **52–55 min** | Better far-face recall |
| GPU, `imgsz=1280`, skip 3 | ~38 | **14–15 min** | Measured YOLO 36.5 ms |
| GPU, `imgsz=1600`, skip 3 | ~27–28 | **20–21 min** | Measured YOLO 66.5 ms |

`video_sample1.mp4` (240 frames): **23–31 s CPU** (Phase 3 CPU build),
**10–13 s GPU** (measured detection cost).

The CPU rows use the CPU-only Torch build measured in Phase 3 (196 ms @1280,
297 ms @1600). The CUDA build's CPU fallback measured **524/552 ms**, so prefer
GPU on this host. GPU rows use measured YOLO latency (36.5/66.5 ms) plus
13 ms/frame decode and ~1 ms amortised embedding; no T4 was available.

These assume sequential file processing with no dropped frames. The pipeline
does not sleep in file mode; it processes as fast as the hardware allows.

---

## 6. Real-time RTSP budget (single 25 fps stream)

### CPU profile (`imgsz=1280`, `frame_skip=4`, measured latency)

| Consumer | ms per second | Share of one core |
|---|---:|---:|
| Decode 25 × 17 ms | 425 | 42.5 % |
| Detect 6.25 × 196 ms | 1 225 | 122.5 % |
| Track 6.25 × 2 ms | 13 | 1.3 % |
| Embedding (~2 new faces/s × 10 ms) | 20 | 2.0 % |
| Event I/O (~0.2 events/s × 15 ms) | 3 | 0.3 % |
| **Total** | **1 686 ms/s** | **≈1.7 cores** |

With `imgsz=1600`, detection rises to **2.47 cores** at skip 3 or
**1.86 cores** at skip 4. On an 8-core machine that is 21–31 % total CPU; on a
4-core machine it is 42–62 %, so use `imgsz=1280` with skip 4–6 there.

**Verdict:** 25 fps RTSP is achievable on 8 cores at `imgsz=1280`, skip 4,
provided decode runs on a separate thread and stale frames are dropped. On a
4-core machine, use `imgsz=1280` with `frame_skip=6` (or `imgsz=960`), or run
file mode. The earlier 90 ms estimate was optimistic by ~2× on this host.

### GPU profile (`imgsz=1600`, skip 3) — measured on RTX 3050 6 GB

| Consumer | ms per second | Share |
|---|---:|---:|
| Decode 25 × 13 ms | 325 | 0.33 core |
| Detect 8.33 × 66.5 ms | 554 | ~55 % GPU |
| Landmarks 2 × 4.1 ms | 8 | ~1 % GPU |
| Embedding 2 × 12.7 ms | 25 | ~2.5 % GPU |
| **GPU utilisation** | — | **~55–60 %** |
| **VRAM** | — | **~2–3 GB** |

At `imgsz=1280`, detection drops to ~36.5 ms per cycle and GPU utilisation is
~30 %. Decode remains ~0.33 CPU core. A 6 GB laptop GPU is sufficient; the
model and CUDA context leave roughly 2–3 GB headroom for other applications.

**Verdict:** 25 fps RTSP is comfortable on the RTX 3050 at 1600/skip 3 and
very comfortable at 1280/skip 3. The previous T4 estimate (10–18 % GPU) was
too optimistic; the measured 55–60 % is the number to plan around.

---

## 7. Memory

| Component | Resident memory |
|---|---:|
| Python + NumPy + OpenCV + PyAV | 180–280 MB |
| ONNX Runtime CPU + YOLO | 250–400 MB |
| InsightFace recognition model | 250–350 MB |
| Frame queue (3 × 2688×1520×3 B = 12.3 MB) | 37 MB |
| Frame queue, 4K (3 × 24.9 MB) | 75 MB |
| Registry, 1 000 faces × 512 × 4 B | 2 MB |
| Registry, 100 000 faces | 205 MB |
| SQLite page cache | 10–50 MB |
| **Typical peak, 2K stream** | **0.9–1.3 GB** |
| **Typical peak, 4K stream** | **1.0–1.5 GB** |
| **GPU VRAM** | **0.8–1.5 GB** (CUDA context + 2 models) |

The frame queue is bounded by design; a slow consumer drops the oldest frame
rather than growing memory.

---

## 8. Storage Growth

### Image sizes

256×256 JPEG, quality 90, face-centred crop: **15–25 KB** per image.
Two images per visit (entry + exit): **30–50 KB/visit**.

### 22-minute sample run

Assume 100–300 face visits:

| Artefact | Size |
|---|---:|
| Entry/exit crops | 3–15 MB |
| SQLite DB (faces + embeddings + visits + events) | 4–10 MB |
| `events.log` (INFO only) | 1–3 MB |
| `debug.log` (per-frame, disabled by default) | 0 |
| **Total** | **< 30 MB** |

### 24/7 RTSP projection (2 000 visitors/day)

| Artefact | Per day | Per 30 days |
|---|---:|---:|
| Crops (4 000 × 20 KB) | 80 MB | 2.4 GB |
| SQLite (events + visits) | 5–8 MB | 150–240 MB |
| `events.log` (rotated 10 MB × 5) | ≤ 10 MB live | capped |
| **Total** | **~90 MB/day** | **~2.7 GB/month** |

Source video is **not** recorded, by design. If full-frame snapshots are added
later, budget ~200 KB each (1600×900 JPEG), which dominates storage.

---

## 9. Where the time actually goes

For the CPU profile, ranked:

1. **YOLO inference** — 55–65 % of processing time. Tune with `imgsz` and
   `frame_skip` first.
2. **HEVC decode** — 25–35 %. Cannot be avoided for RTSP; file mode can
   pre-transcode to 1280-wide H.264 to halve it.
3. **ArcFace embedding** — negligible in steady state because of the per-track
   cache; spikes only when many new faces arrive at once.
4. **I/O** — < 2 %; events are rare.

Optimisation order if the interview machine struggles:

1. `frame_skip` 3 → 4 or 5 (linear detection saving).
2. `imgsz` 1600 → 1280 (≈2× detection speed, loses far faces).
3. `recognition.min_face_size` 60 → 64 (fewer registrations, not faster).
4. Pre-transcode the file dataset to 1280×720 H.264 for demos (halves decode),
   keeping native-resolution config for RTSP.

Do **not** optimise by lowering `min_face_size` or skipping recognition — those
change correctness, not just speed.

---

## 10. Scaling

| Scenario | Requirement |
|---|---|
| 1 RTSP stream, CPU, real-time | ~1.0–1.3 cores; 1 GB RAM |
| 1 RTSP stream, GPU | ~0.5 core + 1.2 GB VRAM; 8–15 % GPU |
| 4 RTSP streams, GPU | ~2 cores decode + 4× VRAM of models (share one model instance, batch if needed) |
| 100 000 registered faces | +205 MB RAM; 3–5 ms match per query |
| 10 million events | DB ~5 GB; add time-based indexes and archive old events |

The design is single-process, single-stream. Multi-stream support would need
one pipeline instance per stream sharing the model objects, or batched
inference. Out of scope.

---

## 11. Benchmark plan (Phase 3 and Phase 8)

`scripts/benchmark.py` will report:

```json
{
  "hardware": {"cpu": "...", "cores": 8, "gpu": "..."},
  "profile": {"imgsz": 1280, "frame_skip": 3},
  "decode_ms": {"p50": 0.0, "p95": 0.0},
  "detect_ms": {"p50": 0.0, "p95": 0.0},
  "embed_ms": {"p50": 0.0, "p95": 0.0},
  "track_ms": {"p50": 0.0, "p95": 0.0},
  "frames": 300,
  "effective_fps": 0.0,
  "peak_rss_mb": 0.0,
  "peak_vram_mb": 0.0
}
```

Procedure:

1. Load one near-face and one far-face frame from `record_20250620_183903.mp4`.
2. Warm up 10 iterations, measure 100.
3. Repeat at `imgsz` = 640, 1280, 1600, 1920.
4. Record measured p50/p95 in a new “Measured on <machine>” section of this
   file.
5. Run the full 22-clip sequence once and record wall time, peak RSS, images
   written, final unique count.

### Measured — Phase 2 (local machine)

PyAV software decode of `record_20250620_183903.mp4` (HEVC, 2688×1520,
1,502 frames):

| Metric | Measured |
|---|---:|
| Decode time | 19.6–20.1 s |
| Decode throughput | **74.8–76.7 fps** |
| Real-time factor | **~3×** |
| Frame shape | `(1520, 2688, 3)` |

Decode is therefore **not** the bottleneck on this machine; YOLO remains the
dominant cost. These numbers replace the decode ranges in §4 for this host and
will be re-measured with detection enabled in Phase 3.

### Measured — Phase 3 (local CPU)

Environment: `torch 2.15.0+cpu`, `ultralytics 8.4.210`, CPU-only. The first
inference after model load includes ~9 s of warmup and is excluded from the
steady-state numbers.

| `imgsz` | front 0 s | front 20 s | front 40 s | 4K crowd | Median ms (2K frame) | Median ms (4K frame) |
|---:|---:|---:|---:|---:|---:|---:|
| 1280 | 3 | 4 | 1 | 11 | 163–196 | 160 |
| 1600 | 3 | 4 | 2 | 13 | 247–311 | 301 |
| 1920 | 3 | 4 | 2 | 13 | 376–417 | 409 |

Observations:

- The checkpoint is **detect-only** (`task=detect`, one class `face`); it emits
  no ArcFace keypoints, so `FaceDetection.keypoints` is `None`.
- Far-corridor faces are still partially missed: 1 face at 40 s with `imgsz`
  1280, 2 at 1600/1920. Optional 2×2 tiling remains the mitigation if Phase 8
  finds recall too low.
- The 4K crowd frame yields 11–13 boxes versus 25–40 visible faces because the
  image is downscaled to the inference size. This is expected; the primary
  dataset is the 2K front-entry camera.
- At `imgsz=1600`, detection alone costs ~2.5 cores at 8.33 cycles/s. The
  original 90 ms estimate in §4 was optimistic by ~2–3× on this host.

### Measured — Phase 4 (RTX 3050 6 GB, CUDA 12.8)

Environment: `torch 2.11.0+cu128`, `onnxruntime-gpu 1.22.0`, CUDA 12.8 with
cuDNN DLLs from the Torch wheel. ONNX Runtime reports
`['CUDAExecutionProvider', 'CPUExecutionProvider']`.

| Stage | GPU median | CPU fallback median |
|---|---:|---:|
| YOLO face @1280 | **36.5 ms** | 524 ms |
| YOLO face @1600 | **66.5 ms** | 552 ms |
| 106-point landmarks | **4.1 ms** | 10.1 ms |
| ArcFace 512-d | **12.7 ms** | 96.6 ms |

Observations:

- The GPU embedding path is ~7.6× faster than CPU for ArcFace and ~2.5× faster
  for landmarks.
- A same-person pair at 0.4 s / 0.8 s scores **0.733 / 0.701** cosine; a
  different person scores **0.118**. The default threshold of 0.45 is
  comfortably between them on this sample.
- `onnxruntime-gpu==1.22.0` is pinned deliberately: 1.31 requests CUDA 13
  DLLs (`cublasLt64_13.dll`), while Torch's cu128 wheel ships CUDA 12.8.
- Most 4K crowd faces fail `blur_threshold=60` because of motion blur; one
  passing face was enough to validate a different-identity embedding. The
  primary 2K front-entry camera is sharper. Phase 8 can tune this threshold if
  registration recall is too low.

### Measured — Phase 5 (local CPU)

ByteTrack (`supervision==0.25.1`) with 10 boxes per detection cycle:

| Metric | Measured |
|---|---:|
| Update median | **1.77 ms** |
| Update p95 | 4.22 ms |

The lifecycle is pure Python over a handful of tracks and visits; its cost is
well below the tracker and does not appear in the frame budget. The budget line
in §4 remains a safe 1–3 ms per cycle.

### Measured — Phase 6 (local disk)

One entry or exit = atomic 256×256 JPEG write + one SQLite transaction:

| Operation | Median | p95 |
|---|---:|---:|
| Entry persistence | **4.78 ms** | 6.36 ms |
| Exit persistence | **4.76 ms** | 5.54 ms |

Events occur a few times per minute, so this is negligible against the frame
budget. Images are fsynced and renamed before the DB commit.

### Measured — Phase 7 (end-to-end smoke)

One 30-frame run from `record_20250620_183903.mp4` on the RTX 3050:

| Metric | Measured |
|---|---:|
| Frames / detection cycles | 30 / 10 |
| Unique faces | 1 |
| Visits / entries / exits / open | 1 / 1 / 1 / 0 |
| Entry crop | 9.3 KB |
| Exit crop | 8.8 KB |
| First event | `2025-06-20T18:39:03.000Z` |

Model load dominates this short run (~10 s); steady-state per-stage costs are
the Phase 3–6 measurements.

### Measured — Phase 8 (full 22-clip pass)

RTX 3050 6 GB, `imgsz=1600`, `frame_skip=3`, `record_*.mp4` sequence:

| Metric | Measured |
|---|---:|
| Clips / frames / detection cycles | 22 / 33,044 / 11,022 |
| Unique faces / visits / anonymous | 26 / 59 / 33 |
| Events / entries / exits / open | 118 / 59 / 59 / 0 |
| Useful wall clock | ~27 min |
| Effective throughput | ~21 fps |
| One clip incl. model load | 83.5 s |
| 4K stress (`video_sample1.mp4`) | 240 frames / 80 cycles, 105.3 s wall; 32 visits, 0 registered |

The 4K crowd clip failed the motion-blur/size gates on every face, so it
produced only anonymous visits; the primary 2K clips registered 26 faces. The
full pass hit one transient GPU stall after 21 clips (CPU idle, GPU spinning).
A fresh process finished the remaining clips in 105.8 s and 194.3 s; the DB was
then verified clean. `nvidia-smi` later reported 510 MHz at 13.5 W, consistent
with a driver/power-state issue rather than a pipeline defect. Mitigation is to
run in shorter segments or restart the process (PLAN R13).

The per-stage measured rows in §6–§9 and the full-run rows above are the
authoritative numbers; the original ranges remain as planning context.

### Docker images (P1, 2026-11-12)

| Target | Size | Notes |
|---|---:|---|
| CPU | 2.23 GB | `python:3.11-slim-bookworm`, uv-synced CPU extras, Torch CPU, ONNX Runtime CPU, headless OpenCV |
| GPU | pending build | CUDA 12.8 runtime base, uv-synced GPU extras, Torch cu128, ONNX Runtime GPU 1.22 |

- Cold CPU build: ~15 minutes on this machine; a source-only rebuild is ~20
  seconds with the dependency layer cached.
- Model volume: 181 MB (6.2 MB YOLO + 174 MB ArcFace); first-start download and
  verification took 101.8 s.
- Container smoke (`imgsz=1280`, 30 frames): 1 unique, 1 entry, 1 exit, 0 open
  on a clean output DB; 29.5 s including model load.
- The image runs as UID 10001, uses `init: true`, `SIGTERM`, and a 30 s stop
  grace period. No model weights, videos, logs or DB are baked in.

---

## 12. Energy and thermal notes

- CPU profile: ~1.7–2.5 cores sustained, package power ~25–45 W on a laptop.
- GPU profile: ~55 % of a 65 W RTX 3050 (~30–40 W) plus CPU decode (~8–15 W).
- Thermal throttling on thin laptops can reduce YOLO throughput by 20–30 %
  over a 20-minute run. If the demo stutters, use file mode or GPU.
