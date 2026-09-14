#!/usr/bin/env fbpython

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path


def _git_output(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_paths(repo: Path, revision: str) -> list[str]:
    output = _git_output(repo, "ls-tree", "-rz", "--name-only", revision)
    return sorted(path.decode("utf-8") for path in output.rstrip(b"\0").split(b"\0"))


def _inventory(repo: Path, revision: str, par: Path) -> dict[str, object]:
    tracked = _tracked_paths(repo, revision)
    with zipfile.ZipFile(par) as archive:
        archive_names = [info.filename for info in archive.infolist()]
        archive_name_set = set(archive_names)
        duplicate_names = sorted(
            name for name, count in Counter(archive_names).items() if count != 1
        )
        entries = []
        for path in tracked:
            if path not in archive_name_set:
                continue
            embedded = archive.read(path)
            committed = _git_output(repo, "show", f"{revision}:{path}")
            embedded_sha256 = _sha256(embedded)
            committed_sha256 = _sha256(committed)
            entries.append(
                {
                    "path": path,
                    "size": len(embedded),
                    "embedded_sha256": embedded_sha256,
                    "committed_sha256": committed_sha256,
                    "match": embedded_sha256 == committed_sha256,
                }
            )

    return {
        "par_path": str(par),
        "par_file_sha256": _file_sha256(par),
        "archive_entry_count": len(archive_names),
        "project_tracked_entry_count": len(entries),
        "duplicate_project_entry_names": [
            name for name in duplicate_names if name in set(tracked)
        ],
        "mismatches": [entry["path"] for entry in entries if not entry["match"]],
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("pars", nargs="+", type=Path)
    args = parser.parse_args()

    full_revision = _git_output(args.repo, "rev-parse", args.revision).decode().strip()
    result = {
        "schema": "embedded_source_inventory_v1",
        "repository": str(args.repo.resolve()),
        "revision": full_revision,
        "archives": [
            _inventory(args.repo, full_revision, par.resolve()) for par in args.pars
        ],
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
