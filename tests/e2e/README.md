# End-to-end tests

`run_e2e.sh` is the CI/local e2e driver. It requires Docker Compose and the
CPU image `intelligent-face-tracker:cpu`.

- `fixtures/entry_exit_3s.mp4` is a 3.04 s / 76-frame trim of the first
  front-entry clip (2688×1520 HEVC, 603 KB).
- `config.file.json` runs the fixture twice. The second run must re-identify
  the Face ID without changing the unique count.
- `config.rtsp.json` runs through MediaMTX and an ffmpeg publisher looping the
  fixture, stopping after 300 frames so active tracks flush as exits.
- `verify_outputs.py` runs inside the app container (UID 10001), opens SQLite,
  and checks counts, integrity, foreign keys, one entry/exit per visit, every
  referenced image and the required `events.log` event types.

Expected file result: 1 unique, 2 visits, 2 entries, 2 exits, 0 open. The RTSP
result is intentionally flexible on visit count but requires 1 unique, at least
one entry/exit, balanced events, 0 open and the full log-event coverage.

```bash
docker build --target cpu -t intelligent-face-tracker:cpu .
bash tests/e2e/run_e2e.sh
```
