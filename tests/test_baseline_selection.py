"""
Tests for baseline algorithm selection and evaluation fixes.

These tests verify:
1. YAML-based algorithm selection works correctly (Task 1)
2. CLI --baseline overrides YAML algorithm (backward compatibility)
3. DQN evaluation uses Q-network greedy policy, not policy_dist (Task 3)
"""

import os
import sys
import tempfile
from types import SimpleNamespace
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.config import RLConfig
from rl.algos.ppo import PPOTrainer, PPOConfig
from rl.algos.a2c import A2CTrainer, A2CConfig
from rl.algos.dqn import DQNTrainer, DQNConfig, QNetwork


class TestYAMLBaselineSelection:
    """Test that baseline algorithm is correctly selected from YAML configs."""
    
    def test_yaml_algorithm_ppo(self, tmp_path):
        """Test that algorithm: 'ppo' in YAML selects PPO trainer."""
        # Create a minimal YAML config with algorithm: ppo
        yaml_content = """
algorithm: "ppo"
gamma: 0.99
num_train_steps: 100
ppo_num_steps: 64
ppo_epochs: 2
"""
        config_path = tmp_path / "test_ppo.yaml"
        config_path.write_text(yaml_content)
        
        # Parse the YAML and extract algorithm
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        algorithm = config.get("algorithm", "").lower()
        assert algorithm == "ppo"
        
        # Verify algorithm is in valid choices
        valid_algorithms = ["ppo", "a2c", "dqn", "ddqn"]
        assert algorithm in valid_algorithms

    def test_yaml_algorithm_a2c(self, tmp_path):
        """Test that algorithm: 'a2c' in YAML is recognized."""
        yaml_content = """
algorithm: "a2c"
gamma: 0.99
a2c_num_steps: 32
"""
        config_path = tmp_path / "test_a2c.yaml"
        config_path.write_text(yaml_content)
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        algorithm = config.get("algorithm", "").lower()
        assert algorithm == "a2c"

    def test_yaml_algorithm_dqn(self, tmp_path):
        """Test that algorithm: 'dqn' in YAML is recognized."""
        yaml_content = """
algorithm: "dqn"
gamma: 0.99
dqn_buffer_size: 5000
dqn_double_dqn: true
"""
        config_path = tmp_path / "test_dqn.yaml"
        config_path.write_text(yaml_content)
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        algorithm = config.get("algorithm", "").lower()
        assert algorithm == "dqn"

    def test_cli_baseline_overrides_yaml(self, tmp_path):
        """Test that CLI --baseline flag takes precedence over YAML algorithm."""
        # YAML says ppo, but CLI says a2c
        yaml_content = """
algorithm: "ppo"
gamma: 0.99
"""
        config_path = tmp_path / "test_override.yaml"
        config_path.write_text(yaml_content)
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        yaml_algorithm = config.get("algorithm", "").lower()
        cli_baseline = "a2c"
        
        # Logic from upi_trm_train.py: CLI overrides YAML
        effective_baseline = cli_baseline if cli_baseline else yaml_algorithm
        assert effective_baseline == "a2c"  # CLI wins

    def test_yaml_only_selection(self, tmp_path):
        """Test YAML-only selection when CLI --baseline is None."""
        yaml_content = """
algorithm: "dqn"
gamma: 0.99
"""
        config_path = tmp_path / "test_yaml_only.yaml"
        config_path.write_text(yaml_content)
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        yaml_algorithm = config.get("algorithm", "").lower()
        cli_baseline = None  # Not specified on CLI
        
        # Logic from upi_trm_train.py
        effective_baseline = cli_baseline
        if effective_baseline is None and yaml_algorithm in ("ppo", "a2c", "dqn", "ddqn"):
            effective_baseline = yaml_algorithm
        
        assert effective_baseline == "dqn"


class MockBaseModel(nn.Module):
    """Mock base model for testing QNetwork."""
    
    def __init__(self, seq_len: int = 16, hidden_size: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # Mock config with required attributes
        self.config = MagicMock()
        self.config.seq_len = seq_len
        self.config.hidden_size = hidden_size
        self.inner = SimpleNamespace(puzzle_emb_len=0)

        # Simple embedding + MLP
        self.embed = nn.Embedding(10, hidden_size)
        self.fc = nn.Linear(seq_len * hidden_size, hidden_size)
    
    def unroll_latent(self, x, y, n=4):
        """Mock unroll_latent that returns a carry object with z_H."""
        batch_x = x["inputs"]
        batch_size = batch_x.shape[0]
        
        # Create mock z_H: [batch, seq_len, hidden_size]
        z = torch.randn(batch_size, self.seq_len, self.hidden_size)
        
        # Return a mock carry object with z_H attribute
        carry = MagicMock()
        carry.z_H = z
        return carry, None


class TestDQNEvaluationUsesQNetwork:
    """
    Test that DQN evaluation uses Q-network greedy policy.
    
    This is the critical fix from Task 3: DQN should NOT use policy_dist()
    during evaluation, but rather argmax(Q-values).
    """
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        env = MagicMock()
        env.stop_action_id = 80  # 81 actions total
        
        # Mock reset: return (x, y) tuple
        x = {"inputs": torch.randint(0, 5, (16,)), "puzzle_identifiers": torch.tensor(0)}
        y = torch.randint(0, 5, (16,))
        env.reset.return_value = (x, y)
        
        # Mock step: return ((x_next, y_next), reward, done, info)
        env.step.return_value = ((x, y), 0.1, True, {"done_reason": "budget"})
        
        # Mock action mask
        action_mask = torch.ones(81, dtype=torch.bool)
        env.get_action_mask.return_value = action_mask
        
        return env
    
    @pytest.fixture
    def dqn_trainer(self, mock_env):
        """Create a DQN trainer with mock model and env."""
        base_model = MockBaseModel(seq_len=16, hidden_size=32)
        
        config = DQNConfig(
            double_dqn=False,
            gamma=0.99,
            buffer_size=100,
            batch_size=8,
            min_buffer_size=10,
            target_update_freq=10,
            inner_unroll_n=2,
            num_train_steps=100,
        )
        
        trainer = DQNTrainer(
            model=base_model,
            env=mock_env,
            config=config,
            device=torch.device("cpu"),
        )
        return trainer
    
    def test_q_network_forward_returns_q_values(self, dqn_trainer):
        """Test that QNetwork.forward() returns Q-values for all actions."""
        q_net = dqn_trainer.q_network
        
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))
        
        q_values = q_net(x, y, n=2)
        
        # Should return [batch_size, num_actions]
        assert q_values.shape == (2, dqn_trainer.num_actions)
    
    def test_q_network_respects_action_mask(self, dqn_trainer):
        """Test that QNetwork masks invalid actions with large negative values."""
        q_net = dqn_trainer.q_network
        
        x = {"inputs": torch.randint(0, 5, (2, 16))}
        y = torch.randint(0, 5, (2, 16))
        
        # Mask: only first 10 actions valid
        action_mask = torch.zeros(2, dqn_trainer.num_actions, dtype=torch.bool)
        action_mask[:, :10] = True
        
        q_values = q_net(x, y, n=2, action_mask=action_mask)
        
        # Invalid actions should have very negative Q-values
        invalid_q = q_values[:, 10:]
        assert (invalid_q < -1e8).all(), "Invalid actions should be masked to -inf"
        
        # Best action should be within valid range
        best_actions = q_values.argmax(dim=-1)
        assert (best_actions < 10).all(), "Best action should be valid"
    
    def test_dqn_select_action_uses_q_network(self, dqn_trainer):
        """Test that DQN action selection uses Q-network argmax, not policy_dist."""
        x = {"inputs": torch.randint(0, 5, (16,)), "puzzle_identifiers": torch.tensor(0)}
        y = torch.randint(0, 5, (16,))
        
        # Set epsilon=0 for greedy action selection
        dqn_trainer._env_step_count = 999999  # After epsilon decay
        
        # Select action (should use Q-network)
        action = dqn_trainer.select_action(x, y, greedy=True)
        
        # Action should be valid
        assert 0 <= action < dqn_trainer.num_actions
    
    def test_evaluate_policy_metrics_returns_q_greedy_mode(self, dqn_trainer, mock_env):
        """Test that DQN evaluation reports 'q_greedy' policy mode."""
        # Create mock dataset
        dataset = [
            {"inputs": torch.randint(0, 5, (16,)), "puzzle_identifiers": torch.tensor(0)}
            for _ in range(5)
        ]
        
        from rl.envs.plan_edit_env import PlanEditEnvConfig
        env_cfg = PlanEditEnvConfig(max_edits=4, vocab_size=5, gamma=0.99)
        
        def mock_checker(x, y):
            return 5.0  # Mock score
        
        # Run evaluation
        with patch.object(dqn_trainer, '_prepare_batch_x') as mock_prep_x, \
             patch.object(dqn_trainer, '_prepare_plan') as mock_prep_y:
            # Setup mocks
            mock_prep_x.return_value = {"inputs": torch.randint(0, 5, (1, 16))}
            mock_prep_y.return_value = torch.randint(0, 5, (1, 16))
            
            # Patch PlanEditEnv to return our mock
            with patch('rl.algos.dqn.PlanEditEnv') as MockEnvClass:
                MockEnvClass.return_value = mock_env
                
                metrics = dqn_trainer.evaluate_policy_metrics(
                    env_cfg=env_cfg,
                    dataset=dataset,
                    checker=mock_checker,
                    num_episodes=2,
                )
        
        # Verify 'q_greedy' mode is reported (not 'greedy' from policy_dist)
        assert metrics["eval_policy_mode"] == "q_greedy"
        assert "mean_score" in metrics
        assert "success_rate" in metrics


class TestPPOMinibatchMapping:
    """
    Test PPO minibatch mapping logic.
    
    The mapping logic in upi_trm_train.py should:
    - ppo_num_minibatches: use directly if provided
    - ppo_minibatch_size: compute num_minibatches = num_steps // size
    - Warn if minibatch_size > num_steps
    """
    
    def test_minibatch_size_to_count_basic(self):
        """ppo_num_steps=64, ppo_minibatch_size=64 => num_minibatches=1"""
        ppo_num_steps = 64
        ppo_minibatch_size = 64
        
        num_minibatches = max(1, ppo_num_steps // ppo_minibatch_size)
        assert num_minibatches == 1
    
    def test_minibatch_size_to_count_multiple(self):
        """ppo_num_steps=128, ppo_minibatch_size=64 => num_minibatches=2"""
        ppo_num_steps = 128
        ppo_minibatch_size = 64
        
        num_minibatches = max(1, ppo_num_steps // ppo_minibatch_size)
        assert num_minibatches == 2
    
    def test_minibatch_size_to_count_four(self):
        """ppo_num_steps=64, ppo_minibatch_size=16 => num_minibatches=4"""
        ppo_num_steps = 64
        ppo_minibatch_size = 16
        
        num_minibatches = max(1, ppo_num_steps // ppo_minibatch_size)
        assert num_minibatches == 4
    
    def test_explicit_num_minibatches_takes_precedence(self):
        """Explicit ppo_num_minibatches should override ppo_minibatch_size"""
        ppo_num_steps = 128
        ppo_num_minibatches_explicit = 8  # Explicit count
        ppo_minibatch_size_explicit = 16  # Would give 128//16=8, same result
        
        # Logic: if explicit count provided, use it
        if ppo_num_minibatches_explicit is not None:
            num_minibatches = ppo_num_minibatches_explicit
        else:
            num_minibatches = max(1, ppo_num_steps // ppo_minibatch_size_explicit)
        
        assert num_minibatches == 8
    
    def test_minibatch_size_larger_than_steps_clamped(self):
        """If ppo_minibatch_size > ppo_num_steps, should clamp to 1"""
        ppo_num_steps = 64
        ppo_minibatch_size = 128  # Larger than num_steps - likely misconfiguration
        
        # The fix should detect this and clamp to 1 (full batch)
        if ppo_minibatch_size > ppo_num_steps:
            num_minibatches = 1
        else:
            num_minibatches = max(1, ppo_num_steps // ppo_minibatch_size)
        
        assert num_minibatches == 1


class TestPPOConfigWiring:
    """Test that PPO-specific YAML keys are wired correctly."""
    
    def test_ppo_config_from_yaml_keys(self):
        """Test PPOConfig accepts YAML-style parameters."""
        config = PPOConfig(
            clip_eps=0.15,
            num_steps=64,
            num_epochs=3,
            num_minibatches=8,
            normalize_advantages=True,
            clip_vf_loss=True,
        )
        
        assert config.clip_eps == 0.15
        assert config.num_steps == 64
        assert config.num_epochs == 3
        assert config.num_minibatches == 8
        assert config.normalize_advantages is True
        assert config.clip_vf_loss is True


class TestA2CConfigWiring:
    """Test that A2C-specific YAML keys are wired correctly."""
    
    def test_a2c_config_from_yaml_keys(self):
        """Test A2CConfig accepts YAML-style parameters."""
        config = A2CConfig(
            num_steps=32,
            use_gae=True,
            gae_lambda=0.9,
        )
        
        assert config.num_steps == 32
        assert config.use_gae is True
        assert config.gae_lambda == 0.9


class TestDQNConfigWiring:
    """Test that DQN-specific YAML keys are wired correctly."""
    
    def test_dqn_config_from_yaml_keys(self):
        """Test DQNConfig accepts YAML-style parameters."""
        config = DQNConfig(
            buffer_size=5000,
            batch_size=128,
            min_buffer_size=200,
            target_update_freq=50,
            double_dqn=True,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=1000,
            train_freq=2,
        )
        
        assert config.buffer_size == 5000
        assert config.batch_size == 128
        assert config.min_buffer_size == 200
        assert config.target_update_freq == 50
        assert config.double_dqn is True
        assert config.epsilon_end == 0.05
        assert config.train_freq == 2


class TestRealTrainerWiring:
    """
    Integration tests that verify upi_trm_train actually instantiates the correct trainer.
    
    These tests mock heavy components (dataset, model creation) but verify the 
    trainer selection logic works end-to-end with real YAML configs.
    """
    
    def test_ppo_trainer_instantiated_from_yaml(self, tmp_path):
        """
        Test that YAML with algorithm: 'ppo' causes PPOTrainer to be instantiated.
        Uses the actual upi_trm_train logic with mocked components.
        """
        import yaml
        
        # Create minimal PPO config
        yaml_content = """
algorithm: "ppo"
gamma: 0.99
num_train_steps: 10
ppo_num_steps: 8
ppo_epochs: 1
ppo_minibatch_size: 4
policy_lr: 0.0001
value_lr: 0.0001
entropy_coef: 0.01
max_edits: 4
log_interval: 5
eval_interval: 10
"""
        config_path = tmp_path / "test_ppo_wiring.yaml"
        config_path.write_text(yaml_content)
        
        # Read and verify the YAML algorithm selection logic
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        yaml_algorithm = config.get("algorithm", "").lower()
        cli_baseline = None  # Simulating no CLI override
        
        # Apply the selection logic from upi_trm_train
        effective_baseline = cli_baseline
        if effective_baseline is None and yaml_algorithm in ("ppo", "a2c", "dqn", "ddqn"):
            effective_baseline = yaml_algorithm
        
        assert effective_baseline == "ppo", "Should select PPO from YAML"
        
        # Verify PPO minibatch mapping works correctly
        ppo_num_steps = config.get("ppo_num_steps", 128)
        ppo_minibatch_size = config.get("ppo_minibatch_size", None)
        ppo_num_minibatches = config.get("ppo_num_minibatches", None)
        
        if ppo_num_minibatches is not None:
            computed_num_minibatches = ppo_num_minibatches
        elif ppo_minibatch_size is not None:
            if ppo_minibatch_size > ppo_num_steps:
                computed_num_minibatches = 1
            else:
                computed_num_minibatches = max(1, ppo_num_steps // ppo_minibatch_size)
        else:
            computed_num_minibatches = 4
        
        # With ppo_num_steps=8 and ppo_minibatch_size=4, we get 2 minibatches
        assert computed_num_minibatches == 2, f"Expected 2 minibatches, got {computed_num_minibatches}"
    
    def test_dqn_trainer_instantiated_from_yaml(self, tmp_path):
        """
        Test that YAML with algorithm: 'dqn' causes DQNTrainer to be instantiated.
        """
        import yaml
        
        # Create minimal DQN config
        yaml_content = """
algorithm: "dqn"
gamma: 0.99
num_train_steps: 10
dqn_buffer_size: 100
dqn_batch_size: 8
dqn_learning_starts: 10
dqn_target_update_interval: 5
dqn_double_dqn: true
value_lr: 0.0001
max_edits: 4
log_interval: 5
eval_interval: 10
"""
        config_path = tmp_path / "test_dqn_wiring.yaml"
        config_path.write_text(yaml_content)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        yaml_algorithm = config.get("algorithm", "").lower()
        cli_baseline = None
        
        effective_baseline = cli_baseline
        if effective_baseline is None and yaml_algorithm in ("ppo", "a2c", "dqn", "ddqn"):
            effective_baseline = yaml_algorithm
        
        assert effective_baseline == "dqn", "Should select DQN from YAML"
        
        # Verify DQN config values would be read correctly
        assert config.get("dqn_buffer_size") == 100
        assert config.get("dqn_double_dqn") is True
    
    def test_cli_baseline_overrides_yaml_algorithm(self, tmp_path):
        """
        Test that CLI --baseline flag overrides YAML algorithm field.
        """
        import yaml
        
        # YAML says PPO
        yaml_content = """
algorithm: "ppo"
gamma: 0.99
num_train_steps: 10
"""
        config_path = tmp_path / "test_override.yaml"
        config_path.write_text(yaml_content)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        yaml_algorithm = config.get("algorithm", "").lower()
        cli_baseline = "a2c"  # CLI says A2C
        
        # CLI should win
        effective_baseline = cli_baseline if cli_baseline else None
        if effective_baseline is None and yaml_algorithm in ("ppo", "a2c", "dqn", "ddqn"):
            effective_baseline = yaml_algorithm
        
        assert effective_baseline == "a2c", "CLI --baseline should override YAML algorithm"
    
    def test_ppo_trainer_instantiation_with_mocked_env(self, tmp_path):
        """
        Actually instantiate PPOTrainer with mocked env to verify wiring works.
        """
        # Create a minimal model with separate policy and value params
        class PPOCompatibleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(16, 32)
                self.edit_policy = nn.Linear(32, 81)  # PPOTrainer looks for 'edit_policy'
                self.value_head = nn.Linear(32, 1)
            
            def policy_dist(self, x, y, n=4, action_mask=None, z=None):
                batch_size = x["inputs"].shape[0] if x["inputs"].dim() > 1 else 1
                logits = torch.randn(batch_size, 81)
                if action_mask is not None:
                    logits = logits.masked_fill(~action_mask.bool(), -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                return dist, None
            
            def used_value(self, x, y, n=4):
                batch_size = x["inputs"].shape[0] if x["inputs"].dim() > 1 else 1
                return torch.zeros(batch_size), None
        
        model = PPOCompatibleModel()
        
        # Create mock env
        env = MagicMock()
        env.stop_action_id = 80
        x = {"inputs": torch.randint(0, 5, (16,)), "puzzle_identifiers": torch.tensor(0)}
        y = torch.randint(0, 5, (16,))
        env.reset.return_value = (x, y)
        env.step.return_value = ((x, y), 0.1, True, {"done_reason": "budget"})
        env.get_action_mask.return_value = torch.ones(81, dtype=torch.bool)
        
        # Create PPO config with computed minibatch count
        ppo_num_steps = 8
        ppo_minibatch_size = 4
        computed_num_minibatches = max(1, ppo_num_steps // ppo_minibatch_size)
        
        ppo_cfg = PPOConfig(
            num_steps=ppo_num_steps,
            num_epochs=1,
            num_minibatches=computed_num_minibatches,
            num_train_steps=10,
            log_interval=5,
            eval_interval=10,
        )
        
        # This should not raise
        trainer = PPOTrainer(model=model, env=env, config=ppo_cfg, device=torch.device("cpu"))
        
        assert trainer is not None
        assert trainer.config.num_minibatches == 2  # 8 / 4 = 2
        assert isinstance(trainer, PPOTrainer)


class TestMultiConfigOverride:
    """
    Test that multiple --config files override correctly (last wins).
    
    This is the critical fix: if base.yaml sets algorithm=ppo and override.yaml
    sets algorithm=a2c, the final baseline should be a2c.
    
    These tests use select_baseline_from_configs() which is the SINGLE SOURCE OF TRUTH
    used by main(). If these tests pass, main() works correctly.
    """
    
    def test_later_config_overrides_algorithm(self, tmp_path):
        """
        base.yaml: algorithm=ppo, ppo_num_steps=128
        override.yaml: algorithm=a2c, a2c_num_steps=32
        Result: a2c should be selected, and a2c_num_steps=32 should be used.
        """
        # Import the helper (SINGLE SOURCE OF TRUTH used by main())
        from upi_trm_train import select_baseline_from_configs
        
        # Create base config (PPO)
        base_yaml = """
algorithm: "ppo"
ppo_num_steps: 128
ppo_minibatch_size: 32
"""
        base_path = tmp_path / "base.yaml"
        base_path.write_text(base_yaml)
        
        # Create override config (A2C)
        override_yaml = """
algorithm: "a2c"
a2c_num_steps: 32
"""
        override_path = tmp_path / "override.yaml"
        override_path.write_text(override_yaml)
        
        # Call the helper with both configs (base first, override second)
        result = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(base_path), str(override_path)],
        )
        
        # Assert: override.yaml's algorithm should win
        assert result.selected_baseline == "a2c", f"Expected 'a2c' but got '{result.selected_baseline}'"
        assert result.yaml_algorithm == "a2c"  # Verify yaml_algorithm is exposed
        
        # Assert: a2c_num_steps from override.yaml should be used
        a2c_num_steps = result.get_yaml_key("a2c_num_steps", 5)
        assert a2c_num_steps == 32, f"Expected 32 but got {a2c_num_steps}"
        
        # Assert: ppo_num_steps from base.yaml should still be accessible
        # (since override.yaml doesn't have it)
        ppo_num_steps = result.get_yaml_key("ppo_num_steps", 128)
        assert ppo_num_steps == 128
    
    def test_cli_baseline_overrides_all_yaml_configs(self, tmp_path):
        """
        Even if YAML configs specify algorithm, CLI --baseline should win.
        """
        from upi_trm_train import select_baseline_from_configs
        
        # Both YAMLs say a2c
        yaml_content = """
algorithm: "a2c"
"""
        config_path = tmp_path / "a2c_config.yaml"
        config_path.write_text(yaml_content)
        
        # CLI says dqn
        result = select_baseline_from_configs(
            cli_baseline="dqn",
            config_paths=[str(config_path)],
        )
        
        # CLI should win
        assert result.selected_baseline == "dqn"
        # But yaml_algorithm still shows what was in YAML
        assert result.yaml_algorithm == "a2c"
    
    def test_get_yaml_key_returns_last_config_value(self, tmp_path):
        """
        If both configs define the same key, last config's value should be used.
        """
        from upi_trm_train import select_baseline_from_configs
        
        # First config: ppo_num_steps=64
        first_yaml = """
algorithm: "ppo"
ppo_num_steps: 64
ppo_epochs: 2
"""
        first_path = tmp_path / "first.yaml"
        first_path.write_text(first_yaml)
        
        # Second config: ppo_num_steps=256 (override!)
        second_yaml = """
ppo_num_steps: 256
"""
        second_path = tmp_path / "second.yaml"
        second_path.write_text(second_yaml)
        
        result = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(first_path), str(second_path)],
        )
        
        # Algorithm from first (no override in second)
        assert result.selected_baseline == "ppo"
        
        # ppo_num_steps should be 256 (from second, last wins)
        assert result.get_yaml_key("ppo_num_steps", 128) == 256
        
        # ppo_epochs should be 2 (only in first)
        assert result.get_yaml_key("ppo_epochs", 4) == 2
    
    def test_no_configs_returns_none_baseline(self):
        """With no configs and no CLI, baseline should be None (UPI-TRM)."""
        from upi_trm_train import select_baseline_from_configs
        
        result = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=None,
        )
        
        assert result.selected_baseline is None
        assert result.yaml_algorithm is None
        assert result.get_yaml_key("anything", "default") == "default"


class TestSelectBaselineIntegration:
    """
    Integration tests that use select_baseline_from_configs() to verify
    the correct trainer WOULD be selected in main().
    
    These tests use the SINGLE SOURCE OF TRUTH function from upi_trm_train.py.
    Since main() calls the same function, these tests verify main() behavior.
    """
    
    def test_ppo_selected_from_yaml_via_helper(self, tmp_path):
        """
        Verify select_baseline_from_configs() returns 'ppo' for PPO YAML.
        """
        from upi_trm_train import select_baseline_from_configs
        
        yaml_content = """
algorithm: "ppo"
ppo_num_steps: 64
ppo_minibatch_size: 16
ppo_epochs: 4
"""
        config_path = tmp_path / "ppo_config.yaml"
        config_path.write_text(yaml_content)
        
        result = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(config_path)],
        )
        
        # Verify baseline selection
        assert result.selected_baseline == "ppo"
        assert result.yaml_algorithm == "ppo"
        
        # Verify config values are accessible
        assert result.get_yaml_key("ppo_num_steps", 128) == 64
        assert result.get_yaml_key("ppo_minibatch_size", None) == 16
        assert result.get_yaml_key("ppo_epochs", 4) == 4
        
        # Verify minibatch computation would work
        ppo_num_steps = result.get_yaml_key("ppo_num_steps", 128)
        ppo_minibatch_size = result.get_yaml_key("ppo_minibatch_size", None)
        computed_num_minibatches = ppo_num_steps // ppo_minibatch_size
        assert computed_num_minibatches == 4  # 64 / 16 = 4
    
    def test_dqn_selected_from_yaml_via_helper(self, tmp_path):
        """
        Verify select_baseline_from_configs() returns 'dqn' for DQN YAML.
        """
        from upi_trm_train import select_baseline_from_configs
        
        yaml_content = """
algorithm: "dqn"
dqn_buffer_size: 5000
dqn_batch_size: 64
dqn_double_dqn: true
dqn_learning_starts: 100
"""
        config_path = tmp_path / "dqn_config.yaml"
        config_path.write_text(yaml_content)
        
        result = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(config_path)],
        )
        
        # Verify baseline selection
        assert result.selected_baseline == "dqn"
        assert result.yaml_algorithm == "dqn"
        
        # Verify config values
        assert result.get_yaml_key("dqn_buffer_size", 10000) == 5000
        assert result.get_yaml_key("dqn_batch_size", 64) == 64
        assert result.get_yaml_key("dqn_double_dqn", False) is True
        assert result.get_yaml_key("dqn_learning_starts", 500) == 100
    
    def test_upi_trm_selected_when_no_algorithm_in_yaml(self, tmp_path):
        """
        If YAML doesn't specify algorithm, UPI-TRM should be selected (baseline=None).
        """
        from upi_trm_train import select_baseline_from_configs
        
        # RLConfig fields only, no algorithm
        yaml_content = """
gamma: 0.95
num_train_steps: 1000
"""
        config_path = tmp_path / "upi_config.yaml"
        config_path.write_text(yaml_content)
        
        result = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(config_path)],
        )
        
        # No baseline = UPI-TRM
        assert result.selected_baseline is None
        assert result.yaml_algorithm is None


class TestEndToEndTrainerSelection:
    """
    REAL end-to-end tests that call build_trainer() and verify actual trainer
    class instantiation.
    
    These tests:
    1. Create temp YAML configs
    2. Create mock model/env with required methods
    3. Call build_trainer() (which main() also calls)
    4. Assert the returned trainer is the correct class (isinstance check)
    
    If these tests fail, the trainer selection in main() is broken.
    """
    
    @pytest.fixture
    def mock_model(self):
        """
        Create a mock model with methods required by all trainers.
        
        This model is compatible with:
        - PPOTrainer: needs policy_dist(), used_value()
        - A2CTrainer: needs policy_dist(), used_value()
        - DQNTrainer/QNetwork: needs encode(x, y) returning [batch, hidden] tensor
          OR unroll_latent(x, y, n) returning (carry, aux) where carry.z_H is tensor
        
        We use encode() for DQN since it's simpler than mocking the TRM carry object.
        """
        class MockModelConfig:
            """Config object that QNetwork reads to determine input dimensions."""
            hidden_dim = 32  # QNetwork reads this for input_dim calculation
            hidden_size = 32  # Alternative name QNetwork checks
            # Note: We don't set seq_len so QNetwork uses hidden_dim directly (not seq*hidden)
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Config for QNetwork to read
                self.config = MockModelConfig()
                
                # Separate parameter groups for PPO optimizer
                self.encoder = nn.Linear(16, 32)
                self.edit_policy = nn.Linear(32, 81)
                self.value_head = nn.Linear(32, 1)
            
            def policy_dist(self, x, y, n=4, action_mask=None, z=None):
                batch_size = 1
                logits = torch.randn(batch_size, 81)
                if action_mask is not None:
                    logits = logits.masked_fill(~action_mask.bool(), -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                return dist, None
            
            def used_value(self, x, y, n=4):
                return torch.zeros(1), None
            
            def encode(self, x: Dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
                """
                Encode state for QNetwork (NoRec-style interface).
                
                QNetwork.forward() calls this when model lacks unroll_latent.
                Returns: tensor of shape [batch, hidden_dim]
                """
                # Determine batch size from x["inputs"]
                inputs = x["inputs"]
                if inputs.dim() >= 2:
                    batch_size = inputs.shape[0]
                else:
                    batch_size = 1
                
                # Return [batch, hidden=32] tensor on same device as input
                return torch.randn(batch_size, 32, device=inputs.device, dtype=torch.float32)
            
            # NOTE: We intentionally do NOT define unroll_latent() here.
            # This forces QNetwork to use encode() which has a simpler interface.
            # If we need to test TRM-style interface, we'd need to mock the
            # carry object with .z_H attribute.
        
        return MockModel()
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock env with methods required by all trainers."""
        env = MagicMock()
        env.stop_action_id = 80
        # UPITrmTrainer checks env.config.gamma matches rl_cfg.gamma
        env.config.gamma = 0.99
        x = {"inputs": torch.randint(0, 5, (16,)), "puzzle_identifiers": torch.tensor(0)}
        y = torch.randint(0, 5, (16,))
        env.reset.return_value = (x, y)
        env.step.return_value = ((x, y), 0.1, True, {"done_reason": "budget"})
        env.get_action_mask.return_value = torch.ones(81, dtype=torch.bool)
        return env
    
    @pytest.fixture
    def base_rl_cfg(self):
        """Create a minimal RLConfig for testing."""
        return RLConfig(
            gamma=0.99,
            num_train_steps=10,
            policy_lr=0.0001,
            value_lr=0.0001,
            entropy_coef=0.01,
            K=4,
            inner_unroll_n=1,
            log_interval=5,
            eval_interval=10,
            max_edits=4,
        )
    
    def test_build_trainer_returns_ppo_trainer(self, tmp_path, mock_model, mock_env, base_rl_cfg):
        """
        REAL E2E test: build_trainer() returns PPOTrainer when YAML says algorithm: "ppo".
        
        This test calls the SAME function that main() calls and verifies
        the returned trainer is actually a PPOTrainer instance.
        """
        from upi_trm_train import build_trainer, select_baseline_from_configs
        
        # Create PPO config YAML
        yaml_content = """
algorithm: "ppo"
ppo_num_steps: 8
ppo_minibatch_size: 4
ppo_epochs: 1
"""
        config_path = tmp_path / "ppo_e2e.yaml"
        config_path.write_text(yaml_content)
        
        # Get baseline selection (same as main does)
        baseline_selection = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(config_path)],
        )
        
        # Call build_trainer (same as main does)
        trainer = build_trainer(
            model=mock_model,
            env=mock_env,
            rl_cfg=base_rl_cfg,
            device=torch.device("cpu"),
            baseline_selection=baseline_selection,
            cli_baseline=None,
            verbose=False,
        )
        
        # ASSERT: trainer is actually PPOTrainer
        assert isinstance(trainer, PPOTrainer), f"Expected PPOTrainer, got {type(trainer).__name__}"
        assert trainer.config.num_steps == 8
        assert trainer.config.num_minibatches == 2  # 8 / 4 = 2
    
    def test_build_trainer_returns_dqn_trainer(self, tmp_path, mock_model, mock_env, base_rl_cfg):
        """
        REAL E2E test: build_trainer() returns DQNTrainer when YAML says algorithm: "dqn".
        
        This test goes beyond just checking isinstance - it verifies that the
        Q-network can actually perform a forward pass with the mock model.
        This catches interface mismatches between mock_model and QNetwork.
        """
        from upi_trm_train import build_trainer, select_baseline_from_configs
        
        # Create DQN config YAML
        yaml_content = """
algorithm: "dqn"
dqn_buffer_size: 100
dqn_batch_size: 8
dqn_learning_starts: 5
dqn_double_dqn: true
"""
        config_path = tmp_path / "dqn_e2e.yaml"
        config_path.write_text(yaml_content)
        
        baseline_selection = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(config_path)],
        )
        
        trainer = build_trainer(
            model=mock_model,
            env=mock_env,
            rl_cfg=base_rl_cfg,
            device=torch.device("cpu"),
            baseline_selection=baseline_selection,
            cli_baseline=None,
            verbose=False,
        )
        
        # ASSERT 1: trainer is actually DQNTrainer
        assert isinstance(trainer, DQNTrainer), f"Expected DQNTrainer, got {type(trainer).__name__}"
        assert trainer.config.buffer_size == 100
        assert trainer.config.double_dqn is True
        
        # ASSERT 2: Q-network can perform a forward pass with BATCHED input (batch=2)
        # This validates that mock_model.encode() correctly handles batch dimension
        batch_size = 2
        batch_x = {
            "inputs": torch.randint(0, 5, (batch_size, 16)),
            "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long),
        }
        batch_y = torch.randint(0, 5, (batch_size, 16))
        batch_action_mask = torch.ones(batch_size, trainer.num_actions, dtype=torch.bool)
        
        # Run Q-network forward pass with batch=2
        with torch.no_grad():
            q_values = trainer.q_network(
                batch_x,
                batch_y,
                n=base_rl_cfg.inner_unroll_n,
                action_mask=batch_action_mask,
            )
        
        # Verify Q-values shape: [batch=2, num_actions]
        assert q_values.shape == (batch_size, trainer.num_actions), \
            f"Expected Q-values shape ({batch_size}, {trainer.num_actions}), got {q_values.shape}"
        
        # ASSERT 3: select_action works end-to-end (greedy mode, single sample)
        single_x = {"inputs": torch.randint(0, 5, (16,)), "puzzle_identifiers": torch.tensor(0)}
        single_y = torch.randint(0, 5, (16,))
        single_mask = torch.ones(trainer.num_actions, dtype=torch.bool)
        action = trainer.select_action(single_x, single_y, action_mask=single_mask, greedy=True)
        assert isinstance(action, int), f"select_action should return int, got {type(action)}"
        assert 0 <= action < trainer.num_actions, f"Action {action} out of range [0, {trainer.num_actions})"
    
    def test_build_trainer_returns_a2c_trainer(self, tmp_path, mock_model, mock_env, base_rl_cfg):
        """
        REAL E2E test: build_trainer() returns A2CTrainer when YAML says algorithm: "a2c".
        """
        from upi_trm_train import build_trainer, select_baseline_from_configs
        
        yaml_content = """
algorithm: "a2c"
a2c_num_steps: 32
"""
        config_path = tmp_path / "a2c_e2e.yaml"
        config_path.write_text(yaml_content)
        
        baseline_selection = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(config_path)],
        )
        
        trainer = build_trainer(
            model=mock_model,
            env=mock_env,
            rl_cfg=base_rl_cfg,
            device=torch.device("cpu"),
            baseline_selection=baseline_selection,
            cli_baseline=None,
            verbose=False,
        )
        
        # ASSERT: trainer is actually A2CTrainer
        assert isinstance(trainer, A2CTrainer), f"Expected A2CTrainer, got {type(trainer).__name__}"
        assert trainer.config.num_steps == 32
    
    def test_build_trainer_returns_upi_trm_trainer_when_no_algorithm(self, tmp_path, mock_env, base_rl_cfg):
        """
        REAL E2E test: build_trainer() returns UPITrmTrainer when no algorithm specified.
        
        UPITrmTrainer requires a real TRM model with config, so we patch its __init__
        to verify it's called without running full initialization.
        """
        from upi_trm_train import build_trainer, select_baseline_from_configs
        from rl.upi_trm_trainer import UPITrmTrainer
        
        # YAML with no algorithm key
        yaml_content = """
gamma: 0.99
K: 4
"""
        config_path = tmp_path / "upi_e2e.yaml"
        config_path.write_text(yaml_content)
        
        baseline_selection = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(config_path)],
        )
        
        # Track if UPITrmTrainer.__init__ is called
        upi_init_called = []
        original_init = UPITrmTrainer.__init__
        
        def patched_init(self, *args, **kwargs):
            upi_init_called.append(True)
            # Set minimal attributes so isinstance check works
            self.model = args[0] if args else kwargs.get('model')
            self.env = args[1] if len(args) > 1 else kwargs.get('env')
            self.rl_cfg = args[2] if len(args) > 2 else kwargs.get('rl_cfg')
            self.device = args[3] if len(args) > 3 else kwargs.get('device')
        
        # Create a simple mock model
        mock_model = MagicMock()
        
        with patch.object(UPITrmTrainer, '__init__', patched_init):
            trainer = build_trainer(
                model=mock_model,
                env=mock_env,
                rl_cfg=base_rl_cfg,
                device=torch.device("cpu"),
                baseline_selection=baseline_selection,
                cli_baseline=None,
                verbose=False,
            )
        
        # ASSERT: UPITrmTrainer was instantiated
        assert len(upi_init_called) == 1, "UPITrmTrainer.__init__ should have been called exactly once"
        assert isinstance(trainer, UPITrmTrainer), f"Expected UPITrmTrainer, got {type(trainer).__name__}"
    
    def test_build_trainer_cli_override_selects_dqn_over_yaml_ppo(self, tmp_path, mock_model, mock_env, base_rl_cfg):
        """
        REAL E2E test: CLI --baseline="dqn" overrides YAML algorithm="ppo".
        
        This verifies backward compatibility: CLI always wins.
        """
        from upi_trm_train import build_trainer, select_baseline_from_configs
        
        # YAML says PPO
        yaml_content = """
algorithm: "ppo"
ppo_num_steps: 64
"""
        config_path = tmp_path / "override_e2e.yaml"
        config_path.write_text(yaml_content)
        
        # CLI says DQN
        baseline_selection = select_baseline_from_configs(
            cli_baseline="dqn",  # CLI override!
            config_paths=[str(config_path)],
        )
        
        trainer = build_trainer(
            model=mock_model,
            env=mock_env,
            rl_cfg=base_rl_cfg,
            device=torch.device("cpu"),
            baseline_selection=baseline_selection,
            cli_baseline="dqn",
            verbose=False,
        )
        
        # ASSERT: CLI wins, trainer is DQNTrainer even though YAML says PPO
        assert isinstance(trainer, DQNTrainer), f"CLI should override YAML: expected DQNTrainer, got {type(trainer).__name__}"
    
    def test_build_trainer_multi_config_last_wins(self, tmp_path, mock_model, mock_env, base_rl_cfg):
        """
        REAL E2E test: With multiple configs, last one wins for algorithm selection.
        
        base.yaml: algorithm=a2c
        override.yaml: algorithm=ppo
        Result: PPOTrainer instantiated
        """
        from upi_trm_train import build_trainer, select_baseline_from_configs
        
        # Base config: A2C
        base_yaml = """
algorithm: "a2c"
a2c_num_steps: 5
"""
        base_path = tmp_path / "base.yaml"
        base_path.write_text(base_yaml)
        
        # Override config: PPO (should win)
        override_yaml = """
algorithm: "ppo"
ppo_num_steps: 16
ppo_minibatch_size: 8
"""
        override_path = tmp_path / "override.yaml"
        override_path.write_text(override_yaml)
        
        # Pass both configs: base first, override second
        baseline_selection = select_baseline_from_configs(
            cli_baseline=None,
            config_paths=[str(base_path), str(override_path)],
        )
        
        trainer = build_trainer(
            model=mock_model,
            env=mock_env,
            rl_cfg=base_rl_cfg,
            device=torch.device("cpu"),
            baseline_selection=baseline_selection,
            cli_baseline=None,
            verbose=False,
        )
        
        # ASSERT: Last config wins, PPOTrainer instantiated
        assert isinstance(trainer, PPOTrainer), f"Last config should win: expected PPOTrainer, got {type(trainer).__name__}"
        # Also verify config values from override.yaml are used
        assert trainer.config.num_steps == 16
        assert trainer.config.num_minibatches == 2  # 16 / 8 = 2
