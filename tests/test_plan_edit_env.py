import torch

from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig


class DummyDataset:
    def __init__(self):
        # Single 1D "puzzle": x is a tensor [3], y is initialized to zeros
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
    dataset = DummyDataset()
    cfg = PlanEditEnvConfig(max_edits=3, gamma=0.99, reward_shaping=True, vocab_size=4)
    env = PlanEditEnv(dataset, dummy_checker, cfg)
    seq_len = dataset.data[0]["inputs"].numel()
    stop_id = seq_len * cfg.vocab_size
    env.set_stop_action_id(stop_id=stop_id)

    x, y = env.reset()
    assert env.step_count == 0
    assert env.done is False

    # Take a non-stop action (edit position 1 -> token 2)
    edit_action = 1 * cfg.vocab_size + 2
    (x1, y1), r1, done1, _ = env.step(action=edit_action)
    assert env.step_count == 1
    assert done1 is False
    assert torch.equal(y1, torch.tensor([0, 2, 0]))

    # Take STOP action
    (x2, y2), r2, done2, _ = env.step(action=stop_id)
    assert done2 is True
    assert env.done is True

    # After STOP, further steps should not be allowed
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
    expected_reward = phi_new + cfg.gamma * phi_new - phi_old
    assert abs(reward - expected_reward) < 1e-6


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
