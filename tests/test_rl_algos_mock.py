
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from rl.algos.a2c import A2CTrainer, A2CConfig
from rl.algos.dqn import DQNTrainer, DQNConfig
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
        config = PPOConfig(num_steps=2, num_epochs=1, num_minibatches=1, inner_unroll_n=0)
        trainer = PPOTrainer(self.model, self.env, config)
        
        # Run one training step
        stats = trainer.train_step()
        
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

if __name__ == "__main__":
    unittest.main()

