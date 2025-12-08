import torch

from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig


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
                "inputs": torch.tensor([2]),
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": torch.tensor([0]),
            }
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def solved_checker(x, y) -> float:
    return 1.0 if torch.equal(y, x["inputs"]) else -1.0


def test_plan_edit_env_step_and_stop():
    """
    Test that:
    1. Edit actions modify the plan correctly
    2. STOP action does NOT terminate (prevents STOP collapse)
    3. Episode terminates when max_edits is reached
    4. After termination, further steps are not allowed
    
    Note: STOP was intentionally changed to NOT terminate the episode
    to prevent the "STOP collapse" issue where the policy learns to
    always stop immediately. The agent should use all available edits.
    """
    dataset = DummyDataset()
    cfg = PlanEditEnvConfig(max_edits=3, gamma=0.99, reward_shaping=True, vocab_size=4)
    env = PlanEditEnv(dataset, dummy_checker, cfg)
    seq_len = dataset.data[0]["inputs"].numel()
    stop_id = seq_len * cfg.vocab_size
    env.set_stop_action_id(stop_id=stop_id)

    x, y = env.reset()
    assert env.step_count == 0
    assert env.done is False

    # Take a non-stop action (edit position 2 -> token 0)
    # The plan starts as inputs.clone() = [1, 2, 3], so we edit position 2 from 3 to 0
    edit_action = 2 * cfg.vocab_size + 0
    (x1, y1), r1, done1, _ = env.step(action=edit_action)
    assert env.step_count == 1
    assert done1 is False
    # Plan is initialized from inputs [1, 2, 3], position 2 edited to 0
    assert torch.equal(y1, torch.tensor([1, 2, 0]))

    # Take STOP action - this now does NOT terminate (to prevent STOP collapse)
    # Instead, it counts as a wasted step with a small penalty
    (x2, y2), r2, done2, info2 = env.step(action=stop_id)
    assert done2 is False, "STOP should not terminate (prevents STOP collapse)"
    assert env.done is False
    assert info2.get("terminated_by_stop") is True, "Should flag that STOP was chosen"
    assert torch.equal(y2, y1), "Plan should be unchanged after STOP"
    
    # Episode terminates when max_edits (3) is reached
    (x3, y3), r3, done3, _ = env.step(action=0)  # Step 3 -> terminates
    assert done3 is True, "Should terminate at max_edits"
    assert env.done is True

    # After termination, further steps should not be allowed
    try:
        env.step(action=0)
        assert False, "Expected an assertion when stepping after episode is done"
    except AssertionError:
        pass


def test_plan_edit_env_terminates_when_solved_threshold_met():
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

    assert done is True
    assert env.done is True
    assert torch.equal(y_next, x["inputs"])

    phi_old = solved_checker(x, y)
    phi_new = solved_checker(x, y_next)
    gamma = cfg.gamma
    # Paper Eq. 4: r = r_0 + γ·Φ(s') - Φ(s)
    # For terminal transitions, r_0 = 0 (no extra terminal bonus in this test)
    expected_reward = gamma * phi_new - phi_old
    assert abs(reward - expected_reward) < 1e-6, f"Expected {expected_reward}, got {reward}"


def test_plan_edit_env_threshold_works_without_reward_shaping():
    dataset = SingleTokenDataset()
    cfg = PlanEditEnvConfig(
        max_edits=2,
        gamma=0.9,
        reward_shaping=False,
        vocab_size=3,
        solved_threshold=0.5,
    )
    env = PlanEditEnv(dataset, solved_checker, cfg)
    seq_len = dataset.data[0]["inputs"].numel()
    env.set_stop_action_id(stop_id=seq_len * cfg.vocab_size)

    env.reset()
    (_, _), reward, done, _ = env.step(action=2)

    assert done is True
    assert abs(reward - 1.0) < 1e-6  # terminal reward equals checker score
