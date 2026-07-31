import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.upi_trm_trainer import compute_k_step_bootstrapped_target


class TinyValueModel(nn.Module):
    """
    A tiny MLP that approximates V(s) on a 1D chain.
    Used to sanity-check that K-step bootstrapped targets move V towards true V^π.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


def _true_value(s: torch.Tensor, gamma: float) -> torch.Tensor:
    vals = []
    for val in s.view(-1):
        n = int(val.item())
        if n <= 0:
            vals.append(0.0)
        else:
            vals.append(float((1.0 - gamma**n) / (1.0 - gamma)))
    return torch.as_tensor(vals, dtype=torch.float32).view_as(s)


def test_k_step_targets_move_value_towards_true_values():
    torch.manual_seed(0)

    gamma = 0.9
    model = TinyValueModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    train_states = torch.arange(0, 5, dtype=torch.float32).view(-1, 1)
    eval_states = train_states[1:]
    true_values = _true_value(eval_states, gamma)
    initial_error = torch.mean(torch.abs(model(eval_states).detach() - true_values))

    for _ in range(200):
        s_next = torch.clamp(train_states - 1.0, min=0.0)
        r = torch.where(train_states > 0, torch.ones_like(train_states), torch.zeros_like(train_states))

        with torch.no_grad():
            v_next = model(s_next).detach()
            td_target = r + gamma * v_next

        v_s = model(train_states)
        loss = F.mse_loss(v_s, td_target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    v_learned = model(eval_states).detach()
    final_error = torch.mean(torch.abs(v_learned - true_values))

    assert final_error < initial_error


def test_k_step_targets_do_not_bootstrap_past_terminal():
    gamma = 0.5
    K = 4

    rewards_K = torch.zeros(1, K, dtype=torch.float32)
    rewards_K[0, 0] = 1.0
    rewards_K[0, 1] = 2.0
    dones_K = torch.zeros(1, K, dtype=torch.bool)
    dones_K[0, 1] = True  # Episode terminates before reaching the full K steps.
    steps_taken = torch.tensor([2], dtype=torch.long)
    v_K = torch.tensor([10.0], dtype=torch.float32)

    targets = compute_k_step_bootstrapped_target(
        rewards_K=rewards_K,
        dones_K=dones_K,
        steps_taken=steps_taken,
        v_K=v_K,
        gamma=gamma,
        K=K,
        exact_k_step_targets=False,
    )

    expected_return = (rewards_K[0, 0] + gamma * rewards_K[0, 1]).item()
    expected_tensor = torch.tensor([expected_return], dtype=targets.dtype)
    assert torch.allclose(targets, expected_tensor, atol=1e-6)


def test_exact_target_rejects_incomplete_nonterminal_segment():
    try:
        compute_k_step_bootstrapped_target(
            rewards_K=torch.tensor([[1.0, 0.0, 0.0]]),
            dones_K=torch.tensor([[False, False, False]]),
            steps_taken=torch.tensor([1]),
            v_K=torch.tensor([2.0]),
            gamma=0.9,
            K=3,
            exact_k_step_targets=True,
        )
    except ValueError as exc:
        assert "requires K transitions" in str(exc)
    else:
        raise AssertionError("Expected incomplete fixed-K segment to fail")
