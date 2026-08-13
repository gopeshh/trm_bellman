"""Unit tests for the unified CleanRL dispatcher."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rl.cleanrl.cleanrl_runner import run


class TestCleanRLRunner(unittest.TestCase):
    def test_dispatches_each_supported_backend(self) -> None:
        cases = (
            (
                {"algo": "ppo"},
                "rl.cleanrl.ppo_trm.run",
            ),
            (
                {"algo": "a2c"},
                "rl.cleanrl.a2c_trm.run",
            ),
            (
                {"algo": "dqn", "env_kind": "cartpole"},
                "rl.cleanrl.dqn_trm.run",
            ),
            (
                {"algo": "dqn", "env_kind": "sudoku", "n_step": 1},
                "rl.cleanrl.dqn_inhouse_wrapper.run",
            ),
        )

        for config, backend in cases:
            with self.subTest(config=config), patch(
                backend,
                return_value={"backend": backend},
            ) as backend_run:
                self.assertEqual(run(config), {"backend": backend})
                backend_run.assert_called_once_with(config)

    def test_rejects_unsupported_sudoku_n_step_before_training(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "wrapper does not yet map n_step",
        ):
            run({"algo": "dqn_n5", "env_kind": "sudoku", "n_step": 5})

    def test_rejects_unknown_algorithm(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            run({"algo": "rainbow"})


if __name__ == "__main__":
    unittest.main()
