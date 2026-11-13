# Submission checklist

**Submitted 2026-11-13.** The repository, public video, and Google Form were
completed. The requirement audit is in [`COMPLIANCE.md`](COMPLIANCE.md).

## Before submitting

- [x] Repository is public: https://github.com/mdaashir/KatomaranHackathon
- [x] `README.md` opens cleanly and the final Katomaran line is present.
- [x] Review [`COMPLIANCE.md`](COMPLIANCE.md) and the `AUTHOR` items.
- [x] Record the walkthrough using [`docs/DEMO.md`](DEMO.md), upload to YouTube,
      and link it from `README.md`, this file and the README bottom.
- [x] Open the video link in a private browser window and confirm it plays.
- [x] `samples/` contains `summary.json`, CSVs, `events.log` and crops.
- [x] Fresh-clone smoke test: setup, download models, run one short clip,
      `--report`, and `scripts/verify_outputs.py`.
- [x] `pytest` passes (235 tests, 90.29% coverage) and the verifier is clean.
- [x] Optional P2 UI smoke: `uv sync --extra cpu --extra ui`, run `uvicorn
      src.api.app:app` + `streamlit run ui/app.py`, and open the dashboard.
- [x] Submit the repository and the public video through the Katomaran
      submission form before **2:00 PM, Friday 13 November 2026**.

## Video

- **Demo video:** [https://youtu.be/bekQvFoOWTY](https://youtu.be/bekQvFoOWTY)
- **Local copy:** [`demo.mkv`](../demo.mkv)
- Script: [`docs/DEMO.md`](DEMO.md)
- Length: 4–5 minutes.

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
