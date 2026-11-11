"""Download and verify the two InsightFace ONNX models used by the tracker.

The full buffalo_l pack is ~275 MB, and the recognizer alone is 174 MB, which
exceeds GitHub's per-file limit. The models are therefore fetched on setup and
kept out of git.
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

BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/"
    "buffalo_l.zip"
)
MODEL_HASHES = {
    "w600k_r50.onnx": (
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
    ),
    "2d106det.onnx": (
        "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf"
    ),
}
DEFAULT_MODEL_DIR = Path("models/insightface/models/buffalo_l")
DEFAULT_ZIP = Path("models/insightface/buffalo_l.zip")


def sha256(path: Path) -> str:
    """Return the hex SHA256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool = False) -> None:
    """Download *url* to *destination* if it is not already present."""

    if destination.is_file() and not force:
        logger.info("using existing archive %s", destination)
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


def extract_models(archive_path: Path, model_dir: Path) -> None:
    """Extract only the recognition and landmark ONNX files."""

    model_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name not in MODEL_HASHES:
                continue
            target = model_dir / name
            logger.info("extracting %s", target)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            found.add(name)

    missing = set(MODEL_HASHES) - found
    if missing:
        raise RuntimeError(
            "archive is missing expected model(s): " + ", ".join(sorted(missing))
        )


def verify_models(model_dir: Path) -> None:
    """Verify every extracted model against its pinned checksum."""

    for name, expected in MODEL_HASHES.items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing model: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"checksum mismatch for {path}: expected {expected}, got {actual}"
            )
        logger.info("verified %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"destination directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=DEFAULT_ZIP,
        help=f"cached archive path (default: {DEFAULT_ZIP})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download the archive even if it exists",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="keep the downloaded archive after extraction",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    download(BUFFALO_L_URL, args.zip, force=args.force)
    extract_models(args.zip, args.model_dir)
    verify_models(args.model_dir)
    if not args.keep_zip:
        args.zip.unlink(missing_ok=True)
        logger.info("removed archive %s", args.zip)
    logger.info("InsightFace models ready in %s", args.model_dir)


if __name__ == "__main__":
    main()
