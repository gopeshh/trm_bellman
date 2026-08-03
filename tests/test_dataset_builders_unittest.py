"""Determinism tests for downloaded dataset builders."""

import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataset import build_easy_sudoku, build_maze_dataset, build_sudoku_dataset
from dataset import build_4x4_sudoku, build_4x4_trivial
from dataset import build_iclr_confirmatory_4x4


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TestDatasetBuilderDeterminism(unittest.TestCase):
    def assert_numpy_rng_state_equal(self, left, right):
        self.assertEqual(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        self.assertEqual(left[2:], right[2:])

    def test_4x4_sudoku_generator_is_local_and_seeded(self):
        random.seed(101)
        np.random.seed(202)
        python_state = random.getstate()
        numpy_state = np.random.get_state()

        first = build_4x4_sudoku.generate_puzzles(12, seed=1729)
        second = build_4x4_sudoku.generate_puzzles(12, seed=1729)
        different = build_4x4_sudoku.generate_puzzles(12, seed=1730)

        self.assertEqual(random.getstate(), python_state)
        self.assert_numpy_rng_state_equal(np.random.get_state(), numpy_state)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(left[1], right[1])
        self.assertTrue(
            any(
                not np.array_equal(left[0], right[0])
                or not np.array_equal(left[1], right[1])
                for left, right in zip(first, different)
            )
        )

    def test_4x4_sudoku_shared_rng_does_not_restart_between_cohorts(self):
        rng = random.Random(42)
        first = build_4x4_sudoku.generate_puzzles(8, 8, 10, rng=rng)
        second = build_4x4_sudoku.generate_puzzles(8, 4, 5, rng=rng)

        self.assertTrue(
            any(
                not np.array_equal(left[1], right[1])
                for left, right in zip(first, second)
            )
        )

    def test_confirmatory_4x4_builder_is_deterministic_and_disjoint(self):
        spec = build_iclr_confirmatory_4x4.BuildSpec(
            splits=(
                build_iclr_confirmatory_4x4.SplitSpec("train", 12, 26080301),
                build_iclr_confirmatory_4x4.SplitSpec(
                    "validation", 6, 26080302
                ),
                build_iclr_confirmatory_4x4.SplitSpec("test", 8, 26080303),
            ),
            producer_commit="a" * 40,
        )
        random.seed(811)
        np.random.seed(812)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first_hashes = build_iclr_confirmatory_4x4.build_dataset(first, spec)
            second_hashes = build_iclr_confirmatory_4x4.build_dataset(second, spec)

            self.assertEqual(_tree_bytes(first), _tree_bytes(second))
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(
                first_hashes,
                build_iclr_confirmatory_4x4.verify_dataset(first),
            )
            for split, count in (("train", 12), ("validation", 6), ("test", 8)):
                manifest = json.loads(
                    (first / "manifests" / f"{split}.json").read_text()
                )
                self.assertEqual(manifest["generated_count"], count)
                self.assertEqual(len(set(manifest["input_sha256s"])), count)
                inputs = np.load(first / split / "all__inputs.npy")
                empty_counts = np.count_nonzero(inputs == 1, axis=1)
                self.assertTrue(np.all((empty_counts >= 6) & (empty_counts <= 8)))

            with self.assertRaisesRegex(
                build_iclr_confirmatory_4x4.DatasetBuildError,
                "Refusing to overwrite",
            ):
                build_iclr_confirmatory_4x4.build_dataset(first, spec)

        self.assertEqual(random.getstate(), python_state)
        self.assert_numpy_rng_state_equal(np.random.get_state(), numpy_state)

    def test_confirmatory_4x4_verifier_detects_mutation(self):
        spec = build_iclr_confirmatory_4x4.BuildSpec(
            splits=(
                build_iclr_confirmatory_4x4.SplitSpec("train", 4, 11),
                build_iclr_confirmatory_4x4.SplitSpec("validation", 2, 12),
                build_iclr_confirmatory_4x4.SplitSpec("test", 3, 13),
            ),
            producer_commit="b" * 40,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            build_iclr_confirmatory_4x4.build_dataset(root, spec)
            path = root / "train" / "all__inputs.npy"
            encoded = bytearray(path.read_bytes())
            encoded[-1] ^= 1
            path.write_bytes(encoded)
            with self.assertRaisesRegex(
                build_iclr_confirmatory_4x4.DatasetBuildError,
                "SHA-256 differs",
            ):
                build_iclr_confirmatory_4x4.verify_dataset(root)

    def test_confirmatory_builder_binds_running_source_to_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in build_iclr_confirmatory_4x4.PRODUCER_SOURCE_PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# unrelated source\n")
            with patch.object(
                build_iclr_confirmatory_4x4,
                "assert_git_files_match_head",
            ), self.assertRaisesRegex(
                build_iclr_confirmatory_4x4.DatasetBuildError,
                "differs from the running binary",
            ):
                build_iclr_confirmatory_4x4._verify_producer_source_matches_runtime(
                    root
                )

    def test_confirmatory_builder_publication_never_replaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / "stage"
            output = root / "output"
            stage.mkdir()
            output.mkdir()
            (stage / "new.txt").write_text("new")
            (output / "old.txt").write_text("old")
            with self.assertRaisesRegex(
                build_iclr_confirmatory_4x4.DatasetBuildError,
                "Refusing to replace",
            ):
                build_iclr_confirmatory_4x4._publish_directory_no_replace(
                    stage,
                    output,
                )

            self.assertEqual((output / "old.txt").read_text(), "old")
            self.assertEqual((stage / "new.txt").read_text(), "new")

    def test_confirmatory_builder_refills_after_duplicate_candidate(self):
        solution = np.array(
            [
                [1, 2, 3, 4],
                [3, 4, 1, 2],
                [2, 1, 4, 3],
                [4, 3, 2, 1],
            ],
            dtype=np.int32,
        )
        first = solution.copy()
        second = solution.copy()
        first.flat[[0, 1, 4, 5, 10, 15]] = 0
        second.flat[[2, 3, 6, 7, 8, 9]] = 0
        with patch.object(
            build_iclr_confirmatory_4x4,
            "generate_solved_4x4",
            return_value=solution,
        ), patch.object(
            build_iclr_confirmatory_4x4,
            "create_puzzle",
            side_effect=[first, first.copy(), second],
        ), patch.object(
            build_iclr_confirmatory_4x4,
            "count_solutions",
            return_value=1,
        ):
            records, stats = build_iclr_confirmatory_4x4._generate_split(
                build_iclr_confirmatory_4x4.SplitSpec("train", 2, 17),
                min_empty_cells=6,
                max_empty_cells=8,
                seen_inputs=set(),
                seen_records=set(),
            )

        self.assertEqual(len(records), 2)
        self.assertEqual(stats["attempts"], 3)
        self.assertEqual(stats["rejected_duplicate_input"], 1)

    def test_confirmatory_builder_is_byte_identical_across_processes(self):
        program = """
import sys
from dataset.build_iclr_confirmatory_4x4 import BuildSpec, SplitSpec, build_dataset

build_dataset(
    sys.argv[1],
    BuildSpec(
        splits=(
            SplitSpec("train", 3, 31),
            SplitSpec("validation", 2, 32),
            SplitSpec("test", 2, 33),
        ),
        producer_commit="c" * 40,
    ),
)
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            for output in (first, second):
                subprocess.run(
                    [sys.executable, "-c", program, str(output)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            self.assertEqual(_tree_bytes(first), _tree_bytes(second))

    def test_4x4_trivial_generator_is_local_and_seeded(self):
        random.seed(303)
        np.random.seed(404)
        python_state = random.getstate()
        numpy_state = np.random.get_state()

        first = build_4x4_trivial.generate_ultra_easy_puzzles(12, seed=99)
        second = build_4x4_trivial.generate_ultra_easy_puzzles(12, seed=99)
        different = build_4x4_trivial.generate_ultra_easy_puzzles(12, seed=100)

        self.assertEqual(random.getstate(), python_state)
        self.assert_numpy_rng_state_equal(np.random.get_state(), numpy_state)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(left[1], right[1])
        self.assertTrue(
            any(
                not np.array_equal(left[0], right[0])
                or not np.array_equal(left[1], right[1])
                for left, right in zip(first, different)
            )
        )

    def test_easy_sudoku_helpers_use_only_the_supplied_generator(self):
        random.seed(405)
        np.random.seed(406)
        python_state = random.getstate()
        numpy_state = np.random.get_state()

        def generate(seed):
            rng = np.random.default_rng(seed)
            solution = build_easy_sudoku.generate_filled_board(rng)
            puzzle = build_easy_sudoku.create_puzzle(solution, 45, rng)
            augmented = build_easy_sudoku.shuffle_sudoku(puzzle, solution, rng)
            return solution, puzzle, augmented

        first = generate(1729)
        second = generate(1729)
        different = generate(1730)

        for left, right in zip(first[:2], second[:2]):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(first[2], second[2]):
            np.testing.assert_array_equal(left, right)
        self.assertTrue(
            any(
                not np.array_equal(left, right)
                for left, right in zip(first[:2] + first[2], different[:2] + different[2])
            )
        )
        self.assertEqual(random.getstate(), python_state)
        self.assert_numpy_rng_state_equal(np.random.get_state(), numpy_state)

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
            random.seed(505)
            np.random.seed(606)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
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

                different_output = root / "different"
                build_sudoku_dataset.preprocess_data(
                    build_sudoku_dataset.DataProcessConfig(
                        output_dir=str(different_output),
                        source_revision="fixture-revision",
                        seed=1730,
                        subsample_size=5,
                        num_aug=2,
                    )
                )

            self.assertEqual(outputs[0], outputs[1])
            self.assertNotEqual(
                {
                    key: value
                    for key, value in outputs[0].items()
                    if key.endswith(".npy")
                },
                {
                    key: value
                    for key, value in _tree_bytes(different_output).items()
                    if key.endswith(".npy")
                },
            )
            self.assertEqual(random.getstate(), python_state)
            self.assert_numpy_rng_state_equal(np.random.get_state(), numpy_state)
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
            random.seed(707)
            np.random.seed(808)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
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

                different_output = root / "different"
                build_maze_dataset.preprocess_data(
                    build_maze_dataset.DataProcessConfig(
                        output_dir=str(different_output),
                        source_revision="fixture-revision",
                        seed=43,
                        subsample_size=3,
                        aug=True,
                    )
                )

            self.assertEqual(outputs[0], outputs[1])
            self.assertNotEqual(
                {
                    key: value
                    for key, value in outputs[0].items()
                    if key.endswith(".npy")
                },
                {
                    key: value
                    for key, value in _tree_bytes(different_output).items()
                    if key.endswith(".npy")
                },
            )
            self.assertEqual(random.getstate(), python_state)
            self.assert_numpy_rng_state_equal(np.random.get_state(), numpy_state)


if __name__ == "__main__":
    unittest.main()
