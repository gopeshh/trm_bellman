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

