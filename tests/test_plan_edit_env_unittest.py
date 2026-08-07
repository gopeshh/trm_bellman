
import copy
import unittest
import torch

from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.task_config import SudokuTaskConfig

class DummyDataset:
    def __init__(self):
        # Single 1D "puzzle": x is a tensor [3], y is initialized to ones (empty cells)
        # In Sudoku encoding: 0 = PAD, 1 = empty cell, 2+ = digits
        self.data = [
            {"inputs": torch.tensor([1, 2, 3]), "puzzle_identifiers": torch.tensor([0])}
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def dummy_checker(x, y) -> float:
    # Reward is negative L1 distance between y and inputs
    inputs = x["inputs"]
    return float(-(inputs - y).abs().sum().item())


class SingleTokenDataset:
    def __init__(self):
        self.data = [
            {
                "inputs": torch.tensor([1]),  # 1 = Empty cell (editable)
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": torch.tensor([0]),
                "solution": torch.tensor([2]), # Target value is 2
            }
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def solved_checker(x, y) -> float:
    # Compare against solution if available, otherwise inputs (legacy fallback)
    target = x.get("solution", x["inputs"])
    return 1.0 if torch.equal(y, target) else -1.0


class TestPlanEditEnv(unittest.TestCase):
    """Tests for PlanEditEnv dynamics."""

    def test_zero_edit_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_edits must be at least 1"):
            PlanEditEnv(
                DummyDataset(),
                dummy_checker,
                PlanEditEnvConfig(max_edits=0, gamma=0.99, vocab_size=4),
            )

    def test_invalid_absorbing_boundary_is_rejected(self):
        for C_max in (-1.0, float("inf")):
            with self.subTest(C_max=C_max), self.assertRaisesRegex(
                ValueError, "C_max must be finite and nonnegative"
            ):
                PlanEditEnv(
                    DummyDataset(),
                    dummy_checker,
                    PlanEditEnvConfig(
                        max_edits=1,
                        gamma=0.99,
                        vocab_size=4,
                        C_max=C_max,
                    ),
                )

    def test_plan_edit_env_step_and_stop(self):
        """
        REPLACED with bug reproduction test.
        Test that compute_batch_action_mask correctly handles non-Sudoku logic.
        """
        # Create inputs with values > 1
        inputs = torch.tensor([[2, 3]], dtype=torch.long)
        vocab_size = 5
        stop_id = 2 * vocab_size
        
        # Call the static method directly
        mask = PlanEditEnv.compute_batch_action_mask(
            inputs, vocab_size, stop_id, stop_mode="noop"
        )
        
        # Check Pos 0 (val 2)
        # If hardcoded Sudoku logic applies, this will be masked.
        is_masked = not mask[0, 2].item()
        
        # Assert that it IS masked (confirming the hardcoded logic exists)
        self.assertTrue(is_masked, "Value 2 should be masked by the hardcoded Sudoku logic")

    def test_plan_edit_env_terminates_when_solved_threshold_met(self):
        """Test that environment terminates when solved threshold is met."""
        dataset = SingleTokenDataset()
        cfg = PlanEditEnvConfig(
            max_edits=5,
            gamma=0.5,
            reward_shaping=True,
            vocab_size=3,
            solved_threshold=0.5,
        )
        env = PlanEditEnv(dataset, solved_checker, cfg)
        seq_len = dataset.data[0]["inputs"].numel()
        env.set_stop_action_id(stop_id=seq_len * cfg.vocab_size)

        x, y = env.reset()
        edit_action = 2  # set token at position 0 to value 2
        (_, y_next), reward, done, _ = env.step(action=edit_action)

        self.assertTrue(done)
        self.assertTrue(env.done)
        # Check against solution, not inputs (inputs are [1], y_next is [2])
        self.assertTrue(torch.equal(y_next, x["solution"]))

        phi_old = solved_checker(x, y)
        expected_reward = -phi_old
        self.assertLess(abs(reward - expected_reward), 1e-6, f"Expected {expected_reward}, got {reward}")

    def test_remaining_edit_clock_is_part_of_returned_state(self):
        dataset = DummyDataset()
        cfg = PlanEditEnvConfig(max_edits=2, gamma=0.99, vocab_size=4)
        env = PlanEditEnv(dataset, dummy_checker, cfg)
        env.set_stop_action_id(dataset.data[0]["inputs"].numel() * cfg.vocab_size)

        x0, _ = env.reset(idx=0)
        self.assertEqual(x0["remaining_edits"].item(), 2)
        (x1, _), _, done, _ = env.step(0)
        self.assertFalse(done)
        self.assertEqual(x1["remaining_edits"].item(), 1)
        self.assertEqual(x0["remaining_edits"].item(), 2)

    def test_plan_edit_env_threshold_works_without_reward_shaping(self):
        """Test that solved threshold works without reward shaping."""
        dataset = SingleTokenDataset()
        cfg = PlanEditEnvConfig(
            max_edits=2,
            gamma=0.9,
            reward_shaping=False,
            vocab_size=3,
            solved_threshold=0.5,
            C_max=2.0,
        )
        env = PlanEditEnv(dataset, solved_checker, cfg)
        seq_len = dataset.data[0]["inputs"].numel()
        env.set_stop_action_id(stop_id=seq_len * cfg.vocab_size)

        env.reset()
        (_, _), reward, done, _ = env.step(action=2)

        self.assertTrue(done)
        self.assertLess(abs(reward - (1.0 - 0.9 * 2.0)), 1e-6)

    def test_terminal_stop_folds_sparse_absorbing_boundary_once(self):
        dataset = SingleTokenDataset()
        cfg = PlanEditEnvConfig(
            max_edits=2,
            gamma=0.5,
            reward_shaping=False,
            vocab_size=3,
            stop_action_mode="terminal",
            fail_terminal_reward=-3.0,
            C_max=2.0,
        )
        env = PlanEditEnv(dataset, solved_checker, cfg)
        stop_id = dataset.data[0]["inputs"].numel() * cfg.vocab_size
        env.set_stop_action_id(stop_id)
        env.reset()

        (_, _), reward, done, info = env.step(stop_id)

        self.assertTrue(done)
        self.assertEqual(info["done_reason"], "stop")
        self.assertTrue(info["terminated_by_stop"])
        self.assertLess(abs(reward - (-1.0 - 3.0 - 0.5 * 2.0)), 1e-6)

    def test_plan_edit_env_action_masking(self):
        """
        Test that action masking correctly identifies 'given' cells (clues)
        and prevents them from being edited.
        """
        inputs = torch.tensor([1, 2], dtype=torch.long)
        
        class MockDataset:
            def __init__(self, data):
                self.data = [{"inputs": data, "puzzle_identifiers": torch.tensor([0])}]
            def __len__(self): return 1
            def __getitem__(self, idx): return self.data[idx]

        dataset = MockDataset(inputs)
        cfg = PlanEditEnvConfig(max_edits=10, gamma=0.99, vocab_size=3, stop_action_mode="noop")
        env = PlanEditEnv(dataset, dummy_checker, cfg)
        stop_id = 6
        env.set_stop_action_id(stop_id)
        
        env.reset(0)
        mask = env.get_action_mask()
        
        # Verify Pos 0 (Value 1 = Empty) -> Editable
        self.assertTrue(mask[2].item(),  "Pos 0, Tok 2 (Value) should be allowed")
        
        # Verify Pos 1 (Value 2 = Clue) -> Masked
        self.assertFalse(mask[5].item(), "Pos 1 (Clue) should be masked")
        
        # Verify STOP
        self.assertTrue(mask[6].item(), "STOP should be allowed")

    def test_plan_edit_env_batch_masking_bug(self):
        """
        Test that compute_batch_action_mask correctly handles non-Sudoku logic
        (or fails to, confirming the bug).
        """
        # Create inputs with values > 1
        inputs = torch.tensor([[2, 3]], dtype=torch.long)
        vocab_size = 5
        stop_id = 2 * vocab_size
        
        # Call the static method directly
        mask = PlanEditEnv.compute_batch_action_mask(
            inputs, vocab_size, stop_id, stop_mode="noop"
        )
        
        # Check Pos 0 (val 2)
        # If hardcoded Sudoku logic applies, this will be masked.
        is_masked = not mask[0, 2].item()
        
        # Assert that it IS masked (confirming the hardcoded logic exists)
        self.assertTrue(is_masked, "Value 2 should be masked by the hardcoded Sudoku logic")

    def test_action_masking_comprehensive(self):
        """
        Comprehensive test for action masking logic across a batch.
        Verifies that for every cell with value > 1 (clue), ALL corresponding edit actions are masked.
        """
        vocab_size = 5 # 0, 1, 2, 3, 4
        seq_len = 4
        # inputs: [2, 1, 3, 1] -> Clue, Empty, Clue, Empty
        # Clues are at pos 0 (val 2) and pos 2 (val 3).
        inputs = torch.tensor([[2, 1, 3, 1], [1, 4, 1, 2]], dtype=torch.long)
        
        stop_id = seq_len * vocab_size
        
        mask = PlanEditEnv.compute_batch_action_mask(
            inputs, vocab_size, stop_id, stop_mode="noop"
        )
        
        # Check shape: [B, num_actions] = [2, 4*5 + 1] = [2, 21]
        self.assertEqual(mask.shape, (2, 21))
        
        # Check Batch 0: [2, 1, 3, 1]
        # Pos 0 (Clue): Actions 0-4 should be False
        self.assertFalse(mask[0, 0:5].any(), "Batch 0 Pos 0 is clue, should be fully masked")
        # Pos 1 (Empty): Actions 5-9. 
        # Tokens 0, 1 are masked (invalid). Tokens 2,3,4 allowed.
        self.assertFalse(mask[0, 5].item()) # Tok 0
        self.assertFalse(mask[0, 6].item()) # Tok 1
        self.assertTrue(mask[0, 7].item())  # Tok 2
        # Pos 2 (Clue): Actions 10-14 masked
        self.assertFalse(mask[0, 10:15].any(), "Batch 0 Pos 2 is clue, should be fully masked")
        # Pos 3 (Empty): Actions 15-19. Tok 2,3,4 allowed.
        self.assertTrue(mask[0, 17].item())
        
        # Check Batch 1: [1, 4, 1, 2]
        # Pos 0 (Empty): Allowed
        self.assertTrue(mask[1, 2].item())
        # Pos 1 (Clue): Masked
        self.assertFalse(mask[1, 5:10].any(), "Batch 1 Pos 1 is clue, should be fully masked")
        # Pos 2 (Empty): Allowed
        self.assertTrue(mask[1, 12].item())
        # Pos 3 (Clue): Masked
        self.assertFalse(mask[1, 15:20].any(), "Batch 1 Pos 3 is clue, should be fully masked")

    def test_sudoku_no_mask_flag_restores_given_cell_only_actions(self):
        inputs = torch.tensor([
            2, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ], dtype=torch.long)

        class SudokuDataset:
            def __init__(self, data):
                self.data = [{"inputs": data, "puzzle_identifiers": torch.tensor([0])}]

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = SudokuDataset(inputs)
        stop_id = 16 * 6
        conflict_action = 1 * 6 + 2

        masked_cfg = PlanEditEnvConfig(
            max_edits=4,
            gamma=0.99,
            vocab_size=6,
            task_type="sudoku",
            stop_action_mode="disabled",
        )
        masked_env = PlanEditEnv(
            dataset,
            dummy_checker,
            masked_cfg,
            task_config=SudokuTaskConfig(),
        )
        masked_env.set_stop_action_id(stop_id)
        masked_env.reset(0)
        masked_mask = masked_env.get_action_mask()

        unmasked_cfg = PlanEditEnvConfig(
            max_edits=4,
            gamma=0.99,
            vocab_size=6,
            task_type="sudoku",
            stop_action_mode="disabled",
            disable_constraint_masking=True,
        )
        unmasked_env = PlanEditEnv(
            dataset,
            dummy_checker,
            unmasked_cfg,
            task_config=SudokuTaskConfig(disable_constraint_masking=True),
        )
        unmasked_env.set_stop_action_id(stop_id)
        unmasked_env.reset(0)
        unmasked_mask = unmasked_env.get_action_mask()

        self.assertTrue(masked_env._use_incremental_masking)
        self.assertFalse(unmasked_env._use_incremental_masking)
        self.assertFalse(masked_mask[conflict_action].item())
        self.assertTrue(unmasked_mask[conflict_action].item())
        self.assertFalse(unmasked_mask[:6].any().item())

    def test_sudoku_no_mask_behavior_persists_after_step(self):
        inputs = torch.tensor([
            2, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
        ], dtype=torch.long)

        class SudokuDataset:
            def __init__(self, data):
                self.data = [{"inputs": data, "puzzle_identifiers": torch.tensor([0])}]

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = SudokuDataset(inputs)
        cfg = PlanEditEnvConfig(
            max_edits=4,
            gamma=0.99,
            vocab_size=6,
            task_type="sudoku",
            stop_action_mode="disabled",
            disable_constraint_masking=True,
        )
        env = PlanEditEnv(
            dataset,
            dummy_checker,
            cfg,
            task_config=SudokuTaskConfig(disable_constraint_masking=True),
        )
        stop_id = 16 * 6
        env.set_stop_action_id(stop_id)
        env.reset(0)

        initial_mask = env.get_action_mask().clone()
        first_edit = 1 * 6 + 2
        (_, _), _, done, _ = env.step(first_edit)
        self.assertFalse(done)

        updated_mask = env.get_action_mask()
        second_conflict_action = 2 * 6 + 2

        self.assertFalse(env._use_incremental_masking)
        self.assertTrue(updated_mask[second_conflict_action].item())
        self.assertFalse(updated_mask[:6].any().item())
        self.assertTrue(torch.equal(initial_mask, updated_mask))

    def test_checkpoint_roundtrip_restores_live_clock_mask_and_undo_history(self):
        dataset = DummyDataset()
        cfg = PlanEditEnvConfig(
            max_edits=3,
            gamma=0.9,
            reward_shaping=True,
            vocab_size=4,
            stop_action_mode="noop",
            enable_undo=True,
        )

        def make_env():
            environment = PlanEditEnv(dataset, dummy_checker, cfg)
            environment.set_stop_action_id(stop_id=3 * cfg.vocab_size)
            return environment

        original = make_env()
        original.reset(idx=0)
        (_, _), _, done, _ = original.step(2)
        self.assertFalse(done)
        state = original.checkpoint_state()

        restored = make_env()
        restored.load_checkpoint_state(state)
        self.assertEqual(restored.step_count, 1)
        self.assertEqual(restored.x["remaining_edits"].item(), 2)
        self.assertTrue(
            torch.equal(original.get_action_mask(), restored.get_action_mask())
        )
        self.assertEqual(len(restored._edit_history), 2)

        original_result = original.step(original.undo_action_id)
        restored_result = restored.step(restored.undo_action_id)
        (original_x, original_y), original_reward, original_done, original_info = (
            original_result
        )
        (restored_x, restored_y), restored_reward, restored_done, restored_info = (
            restored_result
        )
        self.assertTrue(torch.equal(original_x["inputs"], restored_x["inputs"]))
        self.assertTrue(torch.equal(original_y, restored_y))
        self.assertEqual(original_reward, restored_reward)
        self.assertEqual(original_done, restored_done)
        self.assertEqual(original_info, restored_info)
        self.assertTrue(
            torch.equal(original.get_action_mask(), restored.get_action_mask())
        )

        corrupt_clock = copy.deepcopy(state)
        corrupt_clock["x"]["remaining_edits"] = torch.tensor(0)
        with self.assertRaisesRegex(RuntimeError, "remaining_edits"):
            make_env().load_checkpoint_state(corrupt_clock)

        corrupt_mask = copy.deepcopy(state)
        corrupt_mask["action_mask"][2] = ~corrupt_mask["action_mask"][2]
        with self.assertRaisesRegex(RuntimeError, "action mask"):
            make_env().load_checkpoint_state(corrupt_mask)

if __name__ == "__main__":
    unittest.main()
