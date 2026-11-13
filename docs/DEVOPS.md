# Docker, CI & Release Requirements (P1 core, P3 extras)

> **Status (2026-11-12):** P1 Docker/CI/e2e and the P3 security + tagged-release
> workflows are implemented. PR-hygiene, edge publishing and nightly workflows
> were intentionally dropped by the author; Dependabot remains. API/UI e2e
> assertions are deferred until P2 exists.

## 1. Docker

### Files

```
Dockerfile                  # multi-stage: targets `cpu` (default) and `gpu`
entrypoint.sh               # root entrypoint: model bootstrap, then exec
docker-compose.yml          # app (+ optional mediamtx under profile rtsp-test)
docker-compose.gpu.yml      # override: GPU target + reservation
.dockerignore
pyproject.toml  uv.lock    # authoritative dependency groups and lockfile
requirements.txt           # compatibility fallback for pip-only environments
scripts/download_models.py # idempotent; used by the container and by CI
```

### Efficiency rules

- Multi-stage build: a builder stage resolves and builds wheels, the runtime
  stage copies only the venv and the source. `python:3.11-slim` base (match
  the version CI tests).
- Copy dependency manifests first, source last, so code edits don't bust the
  dependency layer. Use BuildKit/uv cache mounts.
- Pinned dependencies in `pyproject.toml` with `cpu`, `gpu` and `ui` extras,
  `dev` in a dependency group, and `uv.lock`; `requirements.txt` is the pip
  fallback only.
- `opencv-python-headless` (no GUI libs). CPU image uses CPU-only PyTorch
  and `onnxruntime`; the GPU target uses a CUDA runtime base, the cu128 Torch
  wheel and `onnxruntime-gpu`.
- Run as a non-root user, `init: true`, `STOPSIGNAL SIGTERM`, and a
  `stop_grace_period: 30s` so the pipeline can flush active tracks as exit
  events and close the DB cleanly. P1 uses a model-readiness healthcheck; P2
  switches it to `/health`.
- Do not bake models, data, logs or clips into the image. Model weights
  (InsightFace, YOLO) download on first start into a **named volume**; the
  download script is idempotent and verifies files exist before starting.
- Record the final image sizes and build time in `docs/COMPUTE.md` and the
  README.

### Compose

- Services: `app` (P1 headless pipeline; P2 API + pipeline, port 8000),
  optional `mediamtx` plus an ffmpeg publisher that loops a sample clip
  (profile `rtsp-test`) for local RTSP testing.
- Volumes: `./data`, `./logs`, `./src/video` bind mounts so the clips and
  outputs are visible on the host; `models` named volume.
- Webcam: run the API on the host that owns the camera. In Linux Docker, pass
  `--device=/dev/video0` to the one-off `docker compose run` command; Docker
  Desktop on Windows does not provide reliable webcam passthrough.
- Config through env vars (`CONFIG_PATH`, `API_URL`, `LOG_LEVEL`, `RTSP_URL`)
  with defaults in compose. No `.env` file is required; secrets are never baked
  into images or printed in logs.
- One-command start: `docker compose up --build`. Headless run (interview
  mode): `docker compose run --rm app python main.py --config config.json`.
  GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`.
- Use `uv` for local dependencies and Docker builds; no Makefile is required.

## 2. CI workflows (`.github/workflows/`)

General rules for every workflow:

- Least-privilege `permissions:` (default `contents: read`; `packages: write`
  only in publish/release jobs), `timeout-minutes` on every job, actions
  pinned to a major version or SHA.
- Cache pip, Docker layers (`type=gha`), and model weights (`actions/cache`
  keyed on model names/versions).
- Upload artifacts on failure and success (logs, crops, DB, e2e report).
- CI needs no secrets. RTSP credentials never appear in workflow logs.
- Hosted runners have no GPU: CI builds the GPU image but only runs the CPU
  one.

### `ci.yml` — verify everything

Triggers: `pull_request` to `main`, **`push` to `main`** (covers direct
commits and merges), `workflow_dispatch`, and `workflow_call` so release and
publish workflows reuse it. `concurrency` cancels superseded PR runs but never
cancels runs on `main`.

| Job | What it does |
|---|---|
| `lint` | `ruff check`, `ruff format --check` |
| `typecheck` | `mypy` (non-blocking at first, blocking once clean) |
| `test` | `pytest` with coverage; fails under a set threshold |
| `build` | `docker buildx build` for cpu, gpu (build only) and ui, no push |
| `e2e` | needs `build`; see below |

**e2e job** (always runs, never path-filtered): brings the stack up with
compose on a short trimmed fixture clip in `tests/fixtures/` and an RTSP
loop via MediaMTX, then runs `tests/e2e/verify_outputs.py`:

- Pipeline exits 0 on the file; unique count equals the expected value in
  `tests/e2e/expected.json`.
- Every face ID has exactly one entry and one exit per visit; exit timestamp
  is not before entry; every event has a row in the DB **and** an image on
  disk.
- `events.log` contains all required event types (detected, embedding
  generated, registered, recognised, tracking, entry, exit).
- Re-running the same clip does not change the unique count (re-ID works).
- RTSP smoke test: stream the looped clip for N seconds, send SIGTERM,
  verify active tracks were flushed as exits and the DB is consistent.
- API `/health` and `/stats` respond; a runtime config override (e.g.
  `frame_skip`) applies and is logged; Streamlit `/_stcore/health` is OK.

Trim the fixture with ffmpeg to a few seconds and a small file size; do not
commit the full provided video.

### `security.yml` — P3

CodeQL (Python), `dependency-review` on PRs, secret scanning with gitleaks
(RTSP credentials are a real risk), `pip-audit`, and Trivy filesystem + built
CPU image scans on every run. Runs on PRs, pushes to `main`, and weekly.

### `release.yml` — P3

Trigger: tag `v*.*.*` (create with `git tag vX.Y.Z && git push --tags`).
Steps: reuse `ci.yml`, build and push versioned images
(`vX.Y.Z`, `X.Y`, `latest`; `-rc` tags are pre-releases), generate release
notes automatically, attach a zip of `samples/` (logs, crops, DB export)
to the GitHub Release.

### `dependabot.yml`

Weekly updates for `uv`, `docker`, and `github-actions`, grouped to keep PR
noise low.

> Edge publishing (`docker-publish.yml`), the PR-hygiene workflow/template and
> `nightly.yml` were dropped by the author to keep the workflow surface small.

## 3. Branch protection (recommended on `main`)

Require the `ci` checks to pass and a PR before merging. The `push` trigger on
`main` stays as the safety net for the author's own direct commits: if it
goes red, fixing it takes priority over new features.

## 4. Build order (one branch and PR each)

1. `chore/dockerfile-and-compose` — headless image, compose, `.dockerignore`,
   root `entrypoint.sh`, `download_models.py`.
2. `ci/core-workflow` — `ci.yml` with lint, test, build (add this as early as
   there is code to test, so later PRs are verified).
3. `test/e2e-fixture-and-verify` — trimmed fixture, `verify_outputs.py`, e2e
   job, RTSP smoke test.
4. `ci/security-and-release` — `security.yml`, `release.yml`, Dependabot.
