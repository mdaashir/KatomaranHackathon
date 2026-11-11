# PROMPTS — key AI prompts used

> Required by `AGENTS.md`: the prompts are evaluated in the interview. This is
> an honest log of the prompts that materially shaped the project, not a
> transcript of every message.

---

## 2026-11-11 — Phase 0: architecture and planning

### Prompt 1 (planning session)

> read @AGENTS.md @"docs\Katomarans Problem Statement.pdf" before start.
> @src\video has video datasets mentioned. Read
> docs/Katomarans_Problem_Statement.pdf. Produce docs/PLAN.md, FEATURES.md,
> COMPUTE.md and ARCHITECTURE.md (Mermaid).

**Purpose:** establish the architecture, feature list, compute budget and
diagrams before writing any code.

**What the model did:**

1. Extracted the PDF with `pdftotext` and read all 143 lines.
2. Probed all 23 videos with `ffprobe`.
3. Extracted representative frames with `ffmpeg` and inspected them visually.
4. Discovered that the local dataset is two different things:
   - 22 × 60 s HEVC clips (2688×1520, 25 fps) of a mall front entry, sparse
     traffic, many back-facing people;
   - one 8 s 4K H.264 clip of a crowded bazaar, suitable as a stress test.
5. Used that evidence to choose native-resolution detection, separate
   track/register size gates, per-track embedding caching, and per-clip
   track flushing.
6. Wrote the four documents and cross-checked terminology, state machine,
   schema and compute profiles for consistency.

**Key follow-up prompts used during the session:**

- “Confirm the entry/exit definition: lifecycle-based or direction-based?”
  → resolved to lifecycle-based (frame appearance → exit timeout), direction
  stored as metadata only.
- “How should back-facing visitors be handled?”
  → documented limitation with an optional person-detector fallback.
- “What happens when the same face reappears after an exit?”
  → new visit, same Face ID, unique count unchanged (no visit reopening).
- “Is `imgsz=1920` real-time on CPU?”
  → no; split into 1600 CPU / 1920 GPU profiles.

**Outputs:** `docs/PLAN.md`, `docs/FEATURES.md`, `docs/COMPUTE.md`,
`docs/ARCHITECTURE.md`.

**Decisions still awaiting author sign-off:** see `PLAN.md §10`.

---

## 2026-11-11 — Phase 1: foundations

### Prompt

> commit the current staged and start proceeding make sure you always
> following the agents.md

**Purpose:** commit the approved planning baseline and start Phase 1 of the
build without crossing the architecture boundaries.

**What the model did:**

1. Restored project context via `/remember restore` (no `memory.md` yet;
   `AGENTS.md` was the only context file).
2. Committed the baseline on `main`, including the docs and project files.
3. Discovered that committing the 232 MB of input `.mp4` files made the push
   fail; removed them from git history, kept them on disk, added
   `src/video/*.mp4` to `.gitignore`, and documented `src/video/README.md`.
4. Built the Phase 1 foundations:
   - `src/config/` — frozen dataclasses, strict loader, `${ENV_VAR}`
     expansion, key-level validation.
   - `src/storage/` — SQLite WAL connection, schema, transaction context,
     repositories for faces, embeddings, visits and events.
   - `src/logging_setup.py` — rotating console/app/debug logs and the
     pipe-delimited `events.log` formatter.
5. Wrote 56 pytest tests and ran the `/review` skill. Fixed the review findings:
   quality range validation, anonymous/linked event consistency, negative
   track IDs, and log field escaping.
6. Pushed `main` and `feature/config-storage-logging`; opened PR #1.

**Key follow-up prompts used during the session:**

- “Why did the push time out?” → auth was fine; the 232 MB input dataset was
  the problem. Removed it from git history.
- “Does the DB enforce one entry and one exit per visit?” → added
  `UNIQUE(visit_id, event_type)` and a duplicate-event test.
- “Are anonymous and linked events allowed to mix?” → no; added validation and
  tests.

**Tests:** `56 passed`.

**Outputs:** `src/config/`, `src/storage/`, `src/logging_setup.py`,
`tests/`, PR #1.

**Open questions:** the five sign-off items in `PLAN.md §10` are still
unresolved; implementation proceeded with the documented defaults.

---

## Template for future phases

```markdown
## YYYY-MM-DD — Phase N: <name>

### Prompt
> <the request, verbatim>

**Purpose:** <why this prompt was used>

**What the model did:**
1. <step>
2. <step>

**Key follow-up prompts:**
- <question> → <resolution>

**Outputs:** <files, branches, PR>

**Corrections made by the author:** <what the author changed and why>
```
