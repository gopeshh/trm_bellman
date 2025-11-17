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


def test_plan_edit_env_step_and_stop():
    dataset = DummyDataset()
    cfg = PlanEditEnvConfig(max_edits=3, gamma=0.99, reward_shaping=True)
    env = PlanEditEnv(dataset, dummy_checker, cfg)
    env.set_stop_action_id(stop_id=0)

    x, y = env.reset()
    assert env.step_count == 0
    assert env.done is False

    # Take a non-stop action (no-op edit by default)
    (x1, y1), r1, done1, _ = env.step(action=1)
    assert env.step_count == 1
    assert done1 is False

    # Take STOP action
    (x2, y2), r2, done2, _ = env.step(action=0)
    assert done2 is True
    assert env.done is True

    # After STOP, further steps should not be allowed
    try:
        env.step(action=0)
        assert False, "Expected an assertion when stepping after episode is done"
    except AssertionError:
        pass

