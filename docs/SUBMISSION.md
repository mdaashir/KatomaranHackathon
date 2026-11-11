# Submission checklist

The code, documentation and samples are complete. The **only manual step left**
is recording the public demo video.

## Before submitting

- [ ] Repository is public: https://github.com/mdaashir/KatomaranHackathon
- [ ] `README.md` opens cleanly and the final Katomaran line is present.
- [ ] Record the walkthrough using [`docs/DEMO.md`](DEMO.md), upload to Loom or
      YouTube, and replace `VIDEO_URL` in `README.md` and this file.
- [ ] Open the video link in a private browser window and confirm it plays.
- [ ] `samples/` contains `summary.json`, CSVs, `events.log` and crops.
- [ ] Fresh-clone smoke test: setup, download models, run one short clip,
      `--report`, and `scripts/verify_outputs.py`.
- [ ] `pytest` passes and the verifier is clean.
- [ ] Submit the repository and the public video through the Katomaran
      submission form.

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
