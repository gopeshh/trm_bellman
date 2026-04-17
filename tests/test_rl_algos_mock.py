
import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from rl.algos.a2c import A2CTrainer, A2CConfig
from rl.algos.dqn import DQNTrainer, DQNConfig, Transition
from rl.algos.ppo import PPOTrainer, PPOConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig

class MockConfig:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        # Add seq_len for QNetwork calculation (it checks config.seq_len)
        self.seq_len = 1 

class MockModel(nn.Module):
    def __init__(self, action_dim=10, hidden_dim=32):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.config = MockConfig(hidden_dim)
        
        # PPO splits params based on "edit_policy" in name
        self.edit_policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.encoder = nn.Linear(10, hidden_dim) 
        
    def policy_dist(self, x, y, n=4, action_mask=None):
        batch_size = x["inputs"].shape[0]
        # Use encoder to get gradients
        z = self.encoder(x["inputs"].float()) # [B, H]
        logits = self.edit_policy_head(z) # [B, A]
        
        if action_mask is not None:
             logits = logits.masked_fill(~action_mask.bool(), -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, None

    def used_value(self, x, y, n=4):
        batch_size = x["inputs"].shape[0]
        z = self.encoder(x["inputs"].float()) # [B, H]
        value = self.value_head(z).squeeze(-1) # [B]
        return value, None
        
    # For DQN (NoRec backbone interface)
    def encode(self, x, y):
        # Return latent
        return self.encoder(x["inputs"].float())

    # For TRM backbone interface (DQN)
    # def unroll_latent(self, x, y, n=4): ... # Not implementing unless needed


class MockTRMModel(nn.Module):
    def __init__(self, hidden_size=8, seq_len=4, puzzle_emb_len=1):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, seq_len=seq_len)
        self.inner = SimpleNamespace(puzzle_emb_len=puzzle_emb_len)
        self.latent_proj = nn.Linear(10, (seq_len + puzzle_emb_len) * hidden_size)

    def unroll_latent(self, x, y, n=4):
        batch_size = x["inputs"].shape[0]
        total_seq_len = self.config.seq_len + self.inner.puzzle_emb_len
        z_flat = self.latent_proj(x["inputs"].float())
        z_H = z_flat.view(batch_size, total_seq_len, self.config.hidden_size)
        return SimpleNamespace(z_H=z_H), None

class TestRLAlgos(unittest.TestCase):
    def setUp(self):
        self.action_dim = 5
        self.stop_action_id = 4
        
        # Mock Environment
        self.env = MagicMock(spec=PlanEditEnv)
        self.env.stop_action_id = self.stop_action_id
        
        # Mock step return: (x, y), reward, done, info
        self.env.step.return_value = (
            ({"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)}, torch.zeros(1, 10)),
            1.0,
            False,
            {}
        )
        # Mock reset return: x, y
        self.env.reset.return_value = (
            {"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)},
            torch.zeros(1, 10)
        )
        # Mock get_action_mask
        self.env.get_action_mask.return_value = torch.ones(self.action_dim, dtype=torch.bool)
        
        self.model = MockModel(action_dim=self.action_dim)

    def test_a2c_train_step(self):
        config = A2CConfig(num_steps=2, inner_unroll_n=0)
        trainer = A2CTrainer(self.model, self.env, config)
        
        # Run one training step
        stats = trainer.train_step()
        
        self.assertIn("loss_policy", stats)
        self.assertIn("loss_value", stats)
        self.assertIn("loss_total", stats)

    def test_ppo_train_step(self):
        config = PPOConfig(num_steps=4, num_epochs=1, num_minibatches=2, inner_unroll_n=0)
        trainer = PPOTrainer(self.model, self.env, config)

        trainer.collect_rollouts(config.num_steps)
        self.assertEqual(torch.stack(trainer.rollout_buffer.log_probs).shape, (config.num_steps,))
        self.assertEqual(torch.stack(trainer.rollout_buffer.values).shape, (config.num_steps,))

        stats = trainer.update()
        
        self.assertIn("loss_policy", stats)
        self.assertIn("loss_value", stats)
        self.assertIn("num_updates", stats)

    def test_dqn_train_step(self):
        config = DQNConfig(
            min_buffer_size=2, # Small buffer to trigger training
            batch_size=2,
            train_freq=1,
            gradient_steps=1,
            inner_unroll_n=0
        )
        trainer = DQNTrainer(self.model, self.env, config)
        
        # Mock env step to return done=True occasionally to test reset
        self.env.step.side_effect = [
             (
                ({"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)}, torch.zeros(1, 10)),
                1.0,
                False,
                {}
            ),
             (
                ({"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)}, torch.zeros(1, 10)),
                1.0,
                True, # Episode done
                {"done_reason": "solved"}
            ),
             (
                ({"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)}, torch.zeros(1, 10)),
                1.0,
                False,
                {}
            )
        ]

        # Run training steps
        # First steps to fill buffer
        trainer.collect_step()
        trainer.collect_step()
        
        # Now train
        stats = trainer.train_step()
        
        self.assertIn("loss_q", stats)
        self.assertIn("mean_q", stats)

    def test_dqn_trm_qnetwork_handles_puzzle_embeddings(self):
        trm_model = MockTRMModel(hidden_size=8, seq_len=4, puzzle_emb_len=2)
        config = DQNConfig(min_buffer_size=1, batch_size=1, inner_unroll_n=0)
        trainer = DQNTrainer(trm_model, self.env, config)

        x, y = self.env.reset.return_value
        q_values = trainer.q_network(
            trainer._prepare_batch_x(x),
            trainer._prepare_plan(y),
            n=config.inner_unroll_n,
            action_mask=self.env.get_action_mask.return_value.to(trainer.device),
        )

        self.assertEqual(q_values.shape, (1, self.action_dim))

    def test_dqn_train_batch_keeps_masks_when_first_transition_is_terminal(self):
        config = DQNConfig(
            min_buffer_size=1,
            batch_size=2,
            train_freq=1,
            gradient_steps=1,
            inner_unroll_n=0,
        )
        trainer = DQNTrainer(self.model, self.env, config)

        x = {"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)}
        y = torch.zeros(1, 10)
        next_mask = torch.tensor([True, False, True, False, False], dtype=torch.bool)

        terminal_transition = Transition(
            x=x,
            y=y,
            action=0,
            reward=1.0,
            x_next=x,
            y_next=y,
            done=True,
            action_mask=torch.ones(self.action_dim, dtype=torch.bool),
            next_action_mask=None,
        )
        masked_transition = Transition(
            x=x,
            y=y,
            action=1,
            reward=1.0,
            x_next=x,
            y_next=y,
            done=False,
            action_mask=torch.ones(self.action_dim, dtype=torch.bool),
            next_action_mask=next_mask,
        )

        trainer.replay_buffer.buffer.clear()
        trainer.replay_buffer.buffer.extend([terminal_transition, masked_transition])
        trainer.replay_buffer.sample = MagicMock(return_value=[terminal_transition, masked_transition])

        online_masks = []
        target_masks = []
        orig_online_forward = trainer.q_network.forward
        orig_target_forward = trainer.target_network.forward

        def wrapped_online_forward(*args, **kwargs):
            online_masks.append(kwargs.get("action_mask"))
            return orig_online_forward(*args, **kwargs)

        def wrapped_target_forward(*args, **kwargs):
            target_masks.append(kwargs.get("action_mask"))
            return orig_target_forward(*args, **kwargs)

        trainer.q_network.forward = wrapped_online_forward
        trainer.target_network.forward = wrapped_target_forward

        stats = trainer.train_batch()

        self.assertIn("loss_q", stats)
        self.assertIsNone(online_masks[0])
        self.assertIsNotNone(online_masks[1])
        self.assertIsNotNone(target_masks[0])
        self.assertEqual(online_masks[1].shape, (2, self.action_dim))
        self.assertTrue(torch.equal(online_masks[1][1].cpu(), next_mask))
        self.assertTrue(torch.equal(target_masks[0][1].cpu(), next_mask))

if __name__ == "__main__":
    unittest.main()
