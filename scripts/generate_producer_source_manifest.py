"""Generate the source manifest embedded in confirmatory training binaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from utils.source_identity import (
    SOURCE_MANIFEST_RELATIVE_PATH,
    build_producer_source_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Repository root to hash. Required when this runs from a packaged "
            "archive, whose link tree is not the repository."
        ),
    )
    arguments = parser.parse_args()
    root = Path(arguments.root).resolve() if arguments.root else _MODULE_ROOT
    if not (root / "utils/source_identity.py").is_file():
        raise SystemExit(f"{root} is not a producer source root.")
    destination = root / SOURCE_MANIFEST_RELATIVE_PATH
    payload = json.dumps(
        build_producer_source_manifest(root),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("ascii") + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
