# Input videos

Place the provided `.mp4` files here. They are **not tracked by git** because
the dataset is ~230 MB of input data; see the ignore rule in the repository
root.

Expected files for the current configuration (`config.json`):

```
record_20250620_183903.mp4
record_20250620_184049.mp4
...
record_20250620_191230.mp4
video_sample1.mp4   # 4K crowded stress-test clip
```

The pipeline reads `record_*.mp4` in filename order and treats each 60-second
clip as an independent observation window. `video_sample1.mp4` is used only for
crowded-scene throughput testing.

Sample outputs produced from these videos are committed under `samples/`.
