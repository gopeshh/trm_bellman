import torch
import torch.nn as nn
import torch.nn.functional as F


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

