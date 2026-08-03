"""Determinism tests for downloaded dataset builders."""

import csv
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dataset import build_easy_sudoku, build_maze_dataset, build_sudoku_dataset
from dataset import build_4x4_sudoku, build_4x4_trivial


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
