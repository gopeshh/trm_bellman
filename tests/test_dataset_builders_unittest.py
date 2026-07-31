"""Determinism tests for downloaded dataset builders."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset import build_maze_dataset, build_sudoku_dataset


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TestDatasetBuilderDeterminism(unittest.TestCase):
    def test_sudoku_builder_is_deterministic_for_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            for split in ("train", "test"):
                with (source_dir / f"{split}.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["source", "question", "answer", "rating"])
                    for index in range(8):
                        question = ["."] * 81
                        question[index] = str((index % 9) + 1)
                        writer.writerow(
                            ["fixture", "".join(question), "123456789" * 9, 10]
                        )

            def fake_download(_repo, filename, **_kwargs):
                return str(source_dir / filename)

            outputs = []
            with patch.object(
                build_sudoku_dataset,
                "hf_hub_download",
                side_effect=fake_download,
            ) as download:
                for name in ("first", "second"):
                    output = root / name
                    build_sudoku_dataset.preprocess_data(
                        build_sudoku_dataset.DataProcessConfig(
                            output_dir=str(output),
                            source_revision="fixture-revision",
                            seed=1729,
                            subsample_size=5,
                            num_aug=2,
                        )
                    )
                    outputs.append(_tree_bytes(output))

            self.assertEqual(outputs[0], outputs[1])
            self.assertTrue(
                all(
                    call.kwargs.get("revision") == "fixture-revision"
                    for call in download.call_args_list
                )
            )

    def test_maze_builder_is_deterministic_for_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            rows = [
                ("S Go", "SGoo"),
                ("S#Go", "S Go"),
                ("S  o", "SGo#"),
                ("S##o", "SG o"),
            ]
            for split in ("train", "test"):
                with (source_dir / f"{split}.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["source", "question", "answer", "rating"])
                    for question, answer in rows:
                        writer.writerow(["fixture", question, answer, 0])

            def fake_download(_repo, filename, **_kwargs):
                return str(source_dir / filename)

            outputs = []
            with patch.object(
                build_maze_dataset,
                "hf_hub_download",
                side_effect=fake_download,
            ):
                for name in ("first", "second"):
                    output = root / name
                    build_maze_dataset.preprocess_data(
                        build_maze_dataset.DataProcessConfig(
                            output_dir=str(output),
                            source_revision="fixture-revision",
                            seed=42,
                            subsample_size=3,
                            aug=True,
                        )
                    )
                    outputs.append(_tree_bytes(output))

            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
