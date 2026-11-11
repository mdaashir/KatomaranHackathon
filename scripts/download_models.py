"""Download and verify all model weights used by the tracker.

No model weights are committed to git. The YOLO face checkpoint is 6.2 MB;
the InsightFace recognizer is 174 MB, which exceeds GitHub's per-file limit.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import shutil
import urllib.request
import zipfile

logger = logging.getLogger("download_models")

YOLO_URL = (
    "https://huggingface.co/deepghs/yolo-face/resolve/main/"
    "yolov8n-face/model.pt"
)
YOLO_SHA256 = (
    "6217b2550b792b789b3e42a55453c6c4326d6293d27f1a6ae56da715c9fbc668"
)

BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/"
    "buffalo_l.zip"
)
INSIGHTFACE_HASHES = {
    "w600k_r50.onnx": (
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
    ),
    "2d106det.onnx": (
        "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"
    ),
}

DEFAULT_MODEL_DIR = Path("models/insightface/models/buffalo_l")
DEFAULT_ZIP = Path("models/insightface/buffalo_l.zip")
DEFAULT_YOLO_PATH = Path("models/yolov8n-face.pt")


def sha256(path: Path) -> str:
    """Return the hex SHA256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches(path: Path, expected: str) -> bool:
    """Return whether *path* exists and has the expected checksum."""

    return path.is_file() and sha256(path) == expected


def download(url: str, destination: Path, force: bool = False) -> None:
    """Download *url* to *destination* if it is not already present."""

    if destination.is_file() and not force:
        logger.info("using existing download %s", destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s -> %s", url, destination)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "intelligent-face-tracker/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def ensure_file(
    url: str,
    destination: Path,
    expected: str,
    *,
    force: bool = False,
) -> None:
    """Ensure a single file exists and matches its pinned checksum."""

    if not force and file_matches(destination, expected):
        logger.info("verified existing %s", destination)
        return
    download(url, destination, force=True)
    if not file_matches(destination, expected):
        raise RuntimeError(
            f"checksum mismatch for {destination}: "
            f"expected {expected}, got {sha256(destination)}"
        )
    logger.info("verified %s", destination)


def extract_models(archive_path: Path, model_dir: Path) -> None:
    """Extract only the recognition and landmark ONNX files."""

    model_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name not in INSIGHTFACE_HASHES:
                continue
            target = model_dir / name
            logger.info("extracting %s", target)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            found.add(name)

    missing = set(INSIGHTFACE_HASHES) - found
    if missing:
        raise RuntimeError(
            "archive is missing expected model(s): " + ", ".join(sorted(missing))
        )


def verify_models(model_dir: Path) -> None:
    """Verify every extracted model against its pinned checksum."""

    for name, expected in INSIGHTFACE_HASHES.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing model: {path}")
        if not file_matches(path, expected):
            raise RuntimeError(
                f"checksum mismatch for {path}: "
                f"expected {expected}, got {sha256(path)}"
            )
        logger.info("verified %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"InsightFace destination (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=DEFAULT_ZIP,
        help=f"cached InsightFace archive (default: {DEFAULT_ZIP})",
    )
    parser.add_argument(
        "--yolo-path",
        type=Path,
        default=DEFAULT_YOLO_PATH,
        help=f"YOLO checkpoint path (default: {DEFAULT_YOLO_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download every artifact and verify it again",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="keep the InsightFace archive after extraction",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ensure_file(
        YOLO_URL,
        args.yolo_path,
        YOLO_SHA256,
        force=args.force,
    )

    insightface_ready = all(
        file_matches(args.model_dir / name, digest)
        for name, digest in INSIGHTFACE_HASHES.items()
    )
    if not insightface_ready or args.force:
        download(BUFFALO_L_URL, args.zip, force=args.force)
        extract_models(args.zip, args.model_dir)
        if not args.keep_zip:
            args.zip.unlink(missing_ok=True)
            logger.info("removed archive %s", args.zip)
    else:
        logger.info("InsightFace models already verified")

    verify_models(args.model_dir)
    logger.info("all models ready")


if __name__ == "__main__":
    main()
