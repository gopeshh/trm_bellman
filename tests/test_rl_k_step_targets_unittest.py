import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.value_targets import compute_k_step_bootstrapped_target


class TinyValueModel(nn.Module):
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


class TestKStepTargets(unittest.TestCase):
    def test_k_step_targets_move_value_towards_true_values(self):
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
        self.assertLess(final_error, initial_error)

    def test_k_step_targets_do_not_bootstrap_past_terminal(self):
        gamma = 0.5
        K = 4

        rewards_K = torch.zeros(1, K, dtype=torch.float32)
        rewards_K[0, 0] = 1.0
        rewards_K[0, 1] = 2.0
        dones_K = torch.zeros(1, K, dtype=torch.bool)
        dones_K[0, 1] = True
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
        self.assertTrue(torch.allclose(targets, expected_tensor, atol=1e-6))

    def test_terminal_reward_already_contains_absorbing_tail(self):
        gamma = 0.9
        K = 3
        rewards_K = torch.tensor([
            [1.0, 2.0, 0.0],  # terminates at step 2
            [1.0, 2.0, 3.0],  # continues
        ], dtype=torch.float32)
        dones_K = torch.tensor([
            [False, True, False],   # done at step 2
            [False, False, False],  # not done
        ], dtype=torch.bool)
        steps_taken = torch.tensor([2, 3], dtype=torch.long)
        # A terminal successor may be outside the model domain. Its nonfinite
        # placeholder must be selected away, not multiplied by zero.
        v_K = torch.tensor([float("nan"), 5.0], dtype=torch.float32)

        targets_with_cmax = compute_k_step_bootstrapped_target(
            rewards_K=rewards_K,
            dones_K=dones_K,
            steps_taken=steps_taken,
            v_K=v_K,
            gamma=gamma,
            K=K,
            exact_k_step_targets=False,
            C_max=10.0,
        )

        targets_without_cmax = compute_k_step_bootstrapped_target(
            rewards_K=rewards_K,
            dones_K=dones_K,
            steps_taken=steps_taken,
            v_K=v_K,
            gamma=gamma,
            K=K,
            exact_k_step_targets=False,
            C_max=None,
        )

        expected_terminal = 1.0 + 0.9 * 2.0
        self.assertAlmostEqual(targets_with_cmax[0].item(), expected_terminal, places=4)
        self.assertAlmostEqual(targets_without_cmax[0].item(), expected_terminal, places=4)
        self.assertTrue(torch.isfinite(targets_with_cmax).all().item())
        self.assertTrue(torch.isfinite(targets_without_cmax).all().item())

        # Sample 1: not terminated, should bootstrap with v_K in both cases
        # G = r0 + γ*r1 + γ^2*r2 + γ^3 * v_K
        expected_1 = 1.0 + 0.9 * 2.0 + (0.9 ** 2) * 3.0 + (0.9 ** 3) * 5.0
        self.assertAlmostEqual(targets_with_cmax[1].item(), expected_1, places=4)
        self.assertAlmostEqual(targets_without_cmax[1].item(), expected_1, places=4)

    def test_exact_target_rejects_incomplete_nonterminal_segment(self):
        with self.assertRaisesRegex(ValueError, "requires K transitions"):
            compute_k_step_bootstrapped_target(
                rewards_K=torch.tensor([[1.0, 0.0, 0.0]]),
                dones_K=torch.tensor([[False, False, False]]),
                steps_taken=torch.tensor([1]),
                v_K=torch.tensor([2.0]),
                gamma=0.9,
                K=3,
                exact_k_step_targets=True,
            )


if __name__ == "__main__":
    unittest.main()
