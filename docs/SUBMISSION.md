# Submission checklist

The code, documentation, samples and optional P2 API/UI are complete. Two
author actions remain: record the public demo video and submit the Google Form.
The requirement audit is in [`COMPLIANCE.md`](COMPLIANCE.md).

## Before submitting

- [ ] Repository is public: https://github.com/mdaashir/KatomaranHackathon
- [ ] `README.md` opens cleanly and the final Katomaran line is present.
- [ ] Review [`COMPLIANCE.md`](COMPLIANCE.md) and the `AUTHOR` items.
- [ ] Record the walkthrough using [`docs/DEMO.md`](DEMO.md), upload to Loom or
      YouTube, and replace `VIDEO_URL` in `README.md` and this file.
- [ ] Open the video link in a private browser window and confirm it plays.
- [ ] `samples/` contains `summary.json`, CSVs, `events.log` and crops.
- [ ] Fresh-clone smoke test: setup, download models, run one short clip,
      `--report`, and `scripts/verify_outputs.py`.
- [ ] `pytest` passes (235 tests, 90.29% coverage) and the verifier is clean.
- [ ] Optional P2 UI smoke: `uv sync --extra cpu --extra ui`, run `uvicorn
      src.api.app:app` + `streamlit run ui/app.py`, and open the dashboard.
- [ ] Submit the repository and the public video through the Katomaran
      submission form before **2:00 PM, Friday 13 November 2026**.

## Video

- **Demo video:** `VIDEO_URL` (replace after recording)
- Script: [`docs/DEMO.md`](DEMO.md)
- Expected length: 4–5 minutes.

## Evidence to mention in the submission

- 22 clips / 33,044 frames.
- 26 unique faces, 59 visits (33 anonymous), 59 entries, 59 exits, 0 open.
- Face 13 re-identified across two clips and counted once.
- 118 events, every one with a cropped image and a SQLite row.
- 4K crowd stress: 32 visits, 0 registered because every face was
  motion-blurred.
- Performance: ~21 fps against a 25 fps source in the long run; tune
  `imgsz`/`frame_skip` on the interview GPU if needed.
- Optional P2 UI: live MJPEG, job queue, folder/RTSP input, validated config,
  visitor gallery and system diagnostics; API/UI are independent of the
  headless pipeline.
