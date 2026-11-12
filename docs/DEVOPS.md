# Docker, CI & Release Requirements (P1 core, P3 extras)

## 1. Docker

### Files

```
Dockerfile                  # multi-stage: targets `cpu` (default) and `gpu`
ui/Dockerfile               # slim image for Streamlit (P2)
docker-compose.yml          # app (+ ui) (+ mediamtx under profile rtsp-test)
docker-compose.gpu.yml      # override: CUDA image + GPU reservation
.dockerignore  .env.example  Makefile
scripts/download_models.py  # idempotent; used by the container and by CI
```

### Efficiency rules

- Multi-stage build: a builder stage resolves and builds wheels, the runtime
  stage copies only the venv and the source. `python:3.11-slim` base (match
  the version CI tests).
- Copy dependency manifests first, source last, so code edits don't bust the
  dependency layer. Use BuildKit cache mounts for pip.
- Pinned, locked dependencies. Split `requirements/` into `base`, `ui`, `dev`
  so the runtime and UI images don't carry test tooling.
- `opencv-python-headless` (no GUI libs). CPU image uses CPU-only PyTorch
  (`--extra-index-url https://download.pytorch.org/whl/cpu`) and
  `onnxruntime`; the GPU target uses a CUDA runtime base and
  `onnxruntime-gpu`.
- Run as a non-root user, `init: true`, `STOPSIGNAL SIGTERM`, a
  `HEALTHCHECK` hitting `/health`, and `stop_grace_period: 30s` so the
  pipeline can flush active tracks as exit events and close the DB cleanly.
- Do not bake models, data, logs or inputs into the image. Model weights
  (InsightFace, YOLO) download on first start into a **named volume**; the
  download script is idempotent and verifies files exist before starting.
- Record the final image sizes and build time in `docs/COMPUTE.md` and the
  README.

### Compose

- Services: `app` (API + pipeline, port 8000), `ui` (Streamlit, port 8501,
  depends on `app` being healthy), optional `mediamtx` plus an ffmpeg
  publisher that loops a sample clip (profile `rtsp-test`) for local RTSP
  testing.
- Volumes: `./data`, `./logs`, `./inputs` bind mounts so outputs are visible on
  the host; `models` named volume.
- Config through env vars (`CONFIG_PATH`, `API_URL`, `LOG_LEVEL`, `RTSP_URL`)
  with `.env.example` committed and `.env` ignored. Secrets are never baked
  into images or printed in logs.
- One-command start: `docker compose up --build`. Headless run (interview
  mode): `docker compose run --rm app python main.py --config config.json`.
  GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`.
- `Makefile` mirrors CI so local and CI behave the same: `make lint test
  build e2e up down`.

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

### `pr-checks.yml` — PR hygiene

- Conventional-commit PR title check (for example
  `amannn/action-semantic-pull-request`).
- Script that fails if any commit in the PR contains a `Co-authored-by` or
  `Claude-Session` trailer (enforces the project's no-attribution rule).
- PR template at `.github/pull_request_template.md` with a checklist: tested,
  e2e passing, docs updated, no attribution trailers.

### `security.yml` — P3

CodeQL (Python), `dependency-review` on PRs, secret scanning with gitleaks
(RTSP credentials are a real risk), `pip-audit`, Trivy scan of the built
image. Runs on PRs, pushes to `main`, and weekly on a schedule.

### `docker-publish.yml` — P3

On push to `main`: reuse `ci.yml`, then build and push images to GHCR with
`edge` and short-SHA tags, buildx cache, and SBOM/provenance.

### `release.yml` — P3

Trigger: tag `v*.*.*` (create with `git tag vX.Y.Z && git push --tags`).
Steps: reuse `ci.yml`, build and push versioned images
(`vX.Y.Z`, `X.Y`, `latest`; `-rc` tags are pre-releases), generate release
notes automatically, attach a zip of `samples/` (logs, crops, DB export)
to the GitHub Release.

### `nightly.yml` — optional

Scheduled full e2e plus security scan against `main` to catch dependency or
model-download drift.

### `dependabot.yml`

Weekly updates for `pip`, `docker`, and `github-actions`, grouped to keep PR
noise low.

## 3. Branch protection (recommended on `main`)

Require the `ci` checks to pass and a PR before merging. The `push` trigger on
`main` stays as the safety net for the author's own direct commits: if it
goes red, fixing it takes priority over new features.

## 4. Build order (one branch and PR each)

1. `chore/dockerfile-and-compose` — headless image, compose, `.dockerignore`,
   `download_models.py`.
2. `ci/core-workflow` — `ci.yml` with lint, test, build (add this as early as
   there is code to test, so later PRs are verified).
3. `test/e2e-fixture-and-verify` — trimmed fixture, `verify_outputs.py`, e2e
   job, RTSP smoke test.
4. `ci/pr-checks` — PR title + trailer checks, PR template, Dependabot.
5. `ci/security-and-release` — `security.yml`, `docker-publish.yml`,
   `release.yml`, optional `nightly.yml`.
