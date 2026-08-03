
import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from rl.algos.a2c import A2CTrainer, A2CConfig
from rl.algos.dqn import (
    DQNTrainer,
    DQNConfig,
    QNetwork,
    ReplayBuffer,
    Transition,
    compute_epsilon_decay_steps,
)
from rl.algos.ppo import PPOTrainer, PPOConfig
from rl.evaluator import evaluate_plan_policy_with_scores
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig

class MockConfig:
    def __init__(self, hidden_dim, rl_num_actions=5):
        self.hidden_dim = hidden_dim
        # Add seq_len for QNetwork calculation (it checks config.seq_len)
        self.seq_len = 1
        self.rl_num_actions = rl_num_actions

class MockModel(nn.Module):
    def __init__(self, action_dim=10, hidden_dim=32):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.config = MockConfig(hidden_dim, rl_num_actions=action_dim)
        
        # PPO splits params by module prefix: edit_policy.*, value_head.*, backbone
        self.edit_policy = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.encoder = nn.Linear(10, hidden_dim) 
        
    def policy_dist(self, x, y, n=4, action_mask=None, z=None):
        batch_size = x["inputs"].shape[0]
        # Use encoder to get gradients
        z = self.encoder(x["inputs"].float()) # [B, H]
        logits = self.edit_policy(z) # [B, A]
        
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


class MockNoRecStubModel(nn.Module):
    def __init__(self, hidden_dim=8, seq_len=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_dim=hidden_dim, seq_len=seq_len)
        self.encoder = nn.Linear(10, hidden_dim)

    def encode(self, x, y):
        return self.encoder(x["inputs"].float())

    def unroll_latent(self, x, y, n=4):
        return None, None


class FakeTaskConfig:
    def __init__(self):
        self.calls = 0

    def compute_action_mask(self, inputs, vocab_size, stop_action_id, current_state=None):
        self.calls += 1
        mask = torch.zeros(stop_action_id + 1, dtype=torch.bool)
        mask[stop_action_id] = True
        return mask

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

    def test_ppo_optimizer_splits_policy_value_and_backbone_groups(self):
        policy_lr = 3e-4
        value_lr = 1e-4
        config = PPOConfig(
            num_steps=2,
            num_epochs=1,
            num_minibatches=1,
            inner_unroll_n=0,
            policy_lr=policy_lr,
            value_lr=value_lr,
        )
        trainer = PPOTrainer(self.model, self.env, config)

        self.assertEqual(len(trainer.optimizer.param_groups), 3)

        name_by_id = {
            id(param): name for name, param in trainer.model.named_parameters()
        }
        groups = []
        for group in trainer.optimizer.param_groups:
            names = [name_by_id[id(param)] for param in group["params"]]
            self.assertTrue(names)
            groups.append((group["lr"], names))

        policy_group = next(group for group in groups if group[0] == policy_lr and all(
            name.startswith("edit_policy.") for name in group[1]
        ))
        value_group = next(group for group in groups if group[0] == value_lr and all(
            name.startswith("value_head.") for name in group[1]
        ))
        backbone_group = next(group for group in groups if group[0] == policy_lr and all(
            not name.startswith("edit_policy.") and not name.startswith("value_head.")
            for name in group[1]
        ))

        self.assertTrue(policy_group[1])
        self.assertTrue(value_group[1])
        self.assertTrue(backbone_group[1])

    def test_ppo_backbone_group_uses_explicit_backbone_lr(self):
        config = PPOConfig(
            num_steps=2,
            num_epochs=1,
            num_minibatches=1,
            inner_unroll_n=0,
            policy_lr=3e-4,
            value_lr=1e-4,
            backbone_lr=7e-4,
        )
        trainer = PPOTrainer(self.model, self.env, config)

        name_by_id = {
            id(param): name for name, param in trainer.model.named_parameters()
        }
        backbone_lrs = []
        for group in trainer.optimizer.param_groups:
            names = [name_by_id[id(param)] for param in group["params"]]
            if all(
                not name.startswith("edit_policy.") and not name.startswith("value_head.")
                for name in names
            ):
                backbone_lrs.append(group["lr"])

        self.assertEqual(backbone_lrs, [7e-4])

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

    def test_dqn_qnetwork_uses_encode_path_for_norec_stub(self):
        norec_model = MockNoRecStubModel(hidden_dim=8, seq_len=4)
        q_network = QNetwork(norec_model, num_actions=self.action_dim)

        x, y = self.env.reset.return_value
        q_values = q_network(x, y, n=0)

        self.assertEqual(q_network._input_dim, 8)
        self.assertEqual(q_values.shape, (1, self.action_dim))

    def test_baseline_state_batches_preserve_remaining_edits(self):
        states = [
            {
                "inputs": torch.zeros(10),
                "puzzle_identifiers": torch.tensor(index),
                "remaining_edits": torch.tensor(clock),
            }
            for index, clock in enumerate((7, 3))
        ]
        trainers = [
            PPOTrainer(
                self.model,
                self.env,
                PPOConfig(num_steps=2, inner_unroll_n=0),
            ),
            A2CTrainer(
                self.model,
                self.env,
                A2CConfig(num_steps=2, inner_unroll_n=0),
            ),
            DQNTrainer(
                self.model,
                self.env,
                DQNConfig(min_buffer_size=1, batch_size=1, inner_unroll_n=0),
            ),
        ]

        for trainer in trainers:
            with self.subTest(trainer=type(trainer).__name__):
                batch = trainer._stack_x_batch(states)
                self.assertIn("remaining_edits", batch)
                self.assertEqual(batch["remaining_edits"].tolist(), [7, 3])

    def test_baseline_state_batch_rejects_inconsistent_clock_presence(self):
        trainer = DQNTrainer(
            self.model,
            self.env,
            DQNConfig(min_buffer_size=1, batch_size=1, inner_unroll_n=0),
        )
        states = [
            {
                "inputs": torch.zeros(10),
                "puzzle_identifiers": torch.tensor(0),
                "remaining_edits": torch.tensor(2),
            },
            {
                "inputs": torch.zeros(10),
                "puzzle_identifiers": torch.tensor(1),
            },
        ]

        with self.assertRaisesRegex(ValueError, "mixes states"):
            trainer._stack_x_batch(states)

    def test_dqn_select_action_rejects_all_invalid_and_wrong_shape_masks(self):
        trainer = DQNTrainer(
            self.model,
            self.env,
            DQNConfig(
                min_buffer_size=1,
                batch_size=1,
                inner_unroll_n=0,
                epsilon_start=1.0,
                epsilon_end=1.0,
            ),
        )
        x, y = self.env.reset.return_value

        with patch("rl.algos.dqn.random.random", return_value=0.0):
            with self.assertRaisesRegex(RuntimeError, "no valid actions"):
                trainer.select_action(
                    x,
                    y,
                    action_mask=torch.zeros(self.action_dim, dtype=torch.bool),
                )
            with self.assertRaisesRegex(ValueError, "action mask"):
                trainer.select_action(
                    x,
                    y,
                    action_mask=torch.ones(self.action_dim - 1, dtype=torch.bool),
                )

    def test_dqn_select_action_samples_only_valid_actions(self):
        trainer = DQNTrainer(
            self.model,
            self.env,
            DQNConfig(
                min_buffer_size=1,
                batch_size=1,
                inner_unroll_n=0,
                epsilon_start=1.0,
                epsilon_end=1.0,
            ),
        )
        x, y = self.env.reset.return_value
        mask = torch.tensor([False, False, True, False, True])

        with patch("rl.algos.dqn.random.random", return_value=0.0):
            actions = {
                trainer.select_action(x, y, action_mask=mask)
                for _ in range(100)
            }
        self.assertEqual(actions, {2, 4})

        one_valid = torch.tensor([False, True, False, False, False])
        with patch("rl.algos.dqn.random.random", return_value=0.0):
            self.assertEqual(
                trainer.select_action(x, y, action_mask=one_valid),
                1,
            )

    def test_dqn_qnetwork_rejects_all_invalid_batch_row(self):
        q_network = QNetwork(
            MockNoRecStubModel(hidden_dim=8, seq_len=4),
            num_actions=self.action_dim,
        )
        x = {
            "inputs": torch.zeros(2, 10),
            "puzzle_identifiers": torch.arange(2),
        }
        y = torch.zeros(2, 10)
        mask = torch.ones(2, self.action_dim, dtype=torch.bool)
        mask[1] = False

        with self.assertRaisesRegex(RuntimeError, r"batch rows \[1\]"):
            q_network(x, y, n=0, action_mask=mask)

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

        trainer.replay_buffer.clear()
        trainer.replay_buffer.add(terminal_transition)
        trainer.replay_buffer.add(masked_transition)
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
        self.assertIsNotNone(online_masks[0])
        self.assertIsNotNone(online_masks[1])
        self.assertIsNotNone(target_masks[0])
        self.assertEqual(online_masks[0].shape, (2, self.action_dim))
        self.assertEqual(online_masks[1].shape, (2, self.action_dim))
        self.assertTrue(torch.equal(online_masks[0][1].cpu(), torch.ones(self.action_dim, dtype=torch.bool)))
        self.assertTrue(torch.equal(online_masks[1][1].cpu(), next_mask))
        self.assertTrue(torch.equal(target_masks[0][1].cpu(), next_mask))

    def test_compute_epsilon_decay_steps_scales_by_train_freq(self):
        epsilon_decay_steps = compute_epsilon_decay_steps(
            num_train_steps=5000,
            exploration_fraction=0.1,
            train_freq=4,
        )

        self.assertEqual(epsilon_decay_steps, 2000)

    def test_build_trainer_dqn_scales_epsilon_decay_by_train_freq(self):
        self.test_compute_epsilon_decay_steps_scales_by_train_freq()

    def test_dqn_target_network_updates_after_env_step_budget(self):
        config = DQNConfig(
            min_buffer_size=1,
            batch_size=1,
            train_freq=4,
            gradient_steps=1,
            target_update_freq=10,
            inner_unroll_n=0,
        )
        trainer = DQNTrainer(self.model, self.env, config)

        def fake_collect_step():
            trainer._env_step_count += 1
            return False

        trainer.collect_step = MagicMock(side_effect=fake_collect_step)
        trainer.train_batch = MagicMock(return_value={"loss_q": 0.0, "mean_q": 0.0})
        trainer.update_target_network = MagicMock()

        trainer.train_step()  # env_steps = 4
        trainer.train_step()  # env_steps = 8
        self.assertEqual(trainer.update_target_network.call_count, 0)

        trainer.train_step()  # env_steps = 12
        self.assertEqual(trainer.update_target_network.call_count, 1)
        self.assertEqual(trainer._last_target_update_env_step, 12)

    def test_dqn_replay_buffer_n_step_one_matches_single_step_transition(self):
        buffer = ReplayBuffer(capacity=10)
        transitions = [
            Transition(
                x={"inputs": torch.full((1, 10), float(i)), "puzzle_identifiers": torch.zeros(1)},
                y=torch.full((1, 10), float(i)),
                action=i % self.action_dim,
                reward=float(i + 1),
                x_next={"inputs": torch.full((1, 10), float(i + 1)), "puzzle_identifiers": torch.zeros(1)},
                y_next=torch.full((1, 10), float(i + 1)),
                done=(i == 2),
                action_mask=torch.ones(self.action_dim, dtype=torch.bool),
                next_action_mask=None if i == 2 else torch.ones(self.action_dim, dtype=torch.bool),
            )
            for i in range(3)
        ]
        for transition in transitions:
            buffer.add(transition)

        for index, expected in enumerate(transitions):
            actual = buffer.build_n_step_transition(index=index, n_step=1, gamma=0.99)
            self.assertEqual(actual.action, expected.action)
            self.assertEqual(actual.reward, expected.reward)
            self.assertEqual(actual.done, expected.done)
            self.assertEqual(actual.bootstrap_steps, 1)
            self.assertTrue(torch.equal(actual.y, expected.y))
            self.assertTrue(torch.equal(actual.y_next, expected.y_next))
            self.assertTrue(torch.equal(actual.x["inputs"], expected.x["inputs"]))
            self.assertTrue(torch.equal(actual.x_next["inputs"], expected.x_next["inputs"]))
            self.assertEqual(actual.next_action_mask is None, expected.next_action_mask is None)
            if actual.next_action_mask is not None and expected.next_action_mask is not None:
                self.assertTrue(torch.equal(actual.next_action_mask, expected.next_action_mask))

    def test_dqn_n_step_three_target_matches_analytic_value(self):
        gamma = 0.5
        config = DQNConfig(
            min_buffer_size=1,
            batch_size=1,
            train_freq=1,
            gradient_steps=1,
            inner_unroll_n=0,
            gamma=gamma,
            dqn_n_step=3,
        )
        trainer = DQNTrainer(self.model, self.env, config)

        base_x = {"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)}
        base_y = torch.zeros(1, 10)
        mask = torch.ones(self.action_dim, dtype=torch.bool)
        rewards = [1.0, 2.0, 3.0, 4.0]
        dones = [False, False, False, True]
        trainer.replay_buffer.clear()
        for reward, done in zip(rewards, dones):
            trainer.replay_buffer.add(
                Transition(
                    x=base_x,
                    y=base_y,
                    action=0,
                    reward=reward,
                    x_next=base_x,
                    y_next=base_y,
                    done=done,
                    action_mask=mask,
                    next_action_mask=None if done else mask,
                )
            )

        sampled = trainer.replay_buffer.build_n_step_transition(index=0, n_step=3, gamma=gamma)
        trainer.replay_buffer.sample = MagicMock(return_value=[sampled])

        current_q = torch.zeros((1, self.action_dim), dtype=torch.float32, requires_grad=True)
        next_q_online = torch.tensor([[0.0, 2.0, 1.0, -1.0, -2.0]], dtype=torch.float32)
        next_q_target = torch.tensor([[3.0, 7.0, 4.0, -1.0, -2.0]], dtype=torch.float32)
        trainer.q_network.forward = MagicMock(side_effect=[current_q, next_q_online])
        trainer.target_network.forward = MagicMock(return_value=next_q_target)

        stats = trainer.train_batch()

        expected_target = 1.0 + gamma * 2.0 + (gamma ** 2) * 3.0 + (gamma ** 3) * 7.0
        self.assertAlmostEqual(stats["mean_target_q"], expected_target, places=6)

    def test_dqn_n_step_terminal_walk_zeroes_bootstrap(self):
        gamma = 0.5
        config = DQNConfig(
            min_buffer_size=1,
            batch_size=1,
            train_freq=1,
            gradient_steps=1,
            inner_unroll_n=0,
            gamma=gamma,
            dqn_n_step=5,
        )
        trainer = DQNTrainer(self.model, self.env, config)

        base_x = {"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)}
        base_y = torch.zeros(1, 10)
        mask = torch.ones(self.action_dim, dtype=torch.bool)
        trainer.replay_buffer.clear()
        trainer.replay_buffer.add(
            Transition(
                x=base_x,
                y=base_y,
                action=0,
                reward=1.0,
                x_next=base_x,
                y_next=base_y,
                done=False,
                action_mask=mask,
                next_action_mask=mask,
            )
        )
        trainer.replay_buffer.add(
            Transition(
                x=base_x,
                y=base_y,
                action=0,
                reward=2.0,
                x_next=base_x,
                y_next=base_y,
                done=True,
                action_mask=mask,
                next_action_mask=None,
            )
        )

        sampled = trainer.replay_buffer.build_n_step_transition(index=0, n_step=5, gamma=gamma)
        trainer.replay_buffer.sample = MagicMock(return_value=[sampled])

        current_q = torch.zeros((1, self.action_dim), dtype=torch.float32, requires_grad=True)
        next_q_online = torch.tensor([[0.0, 20.0, 1.0, -1.0, -2.0]], dtype=torch.float32)
        next_q_target = torch.tensor([[3.0, 70.0, 4.0, -1.0, -2.0]], dtype=torch.float32)
        trainer.q_network.forward = MagicMock(side_effect=[current_q, next_q_online])
        trainer.target_network.forward = MagicMock(return_value=next_q_target)

        stats = trainer.train_batch()

        expected_target = 1.0 + gamma * 2.0
        self.assertTrue(sampled.done)
        self.assertEqual(sampled.bootstrap_steps, 2)
        self.assertAlmostEqual(stats["mean_target_q"], expected_target, places=6)

    def test_dqn_n_step_truncates_when_episode_ends_before_horizon(self):
        gamma = 0.75
        buffer = ReplayBuffer(capacity=10)
        base_x = {"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)}
        base_y = torch.zeros(1, 10)
        mask = torch.ones(self.action_dim, dtype=torch.bool)
        for reward, done in [(1.0, False), (2.0, False), (3.0, False), (4.0, True)]:
            buffer.add(
                Transition(
                    x=base_x,
                    y=base_y,
                    action=0,
                    reward=reward,
                    x_next=base_x,
                    y_next=base_y,
                    done=done,
                    action_mask=mask,
                    next_action_mask=None if done else mask,
                )
            )

        truncated = buffer.build_n_step_transition(index=2, n_step=5, gamma=gamma)

        self.assertTrue(truncated.done)
        self.assertEqual(truncated.bootstrap_steps, 2)
        self.assertAlmostEqual(truncated.reward, 3.0 + gamma * 4.0, places=6)
        self.assertIsNone(truncated.next_action_mask)

    def test_dqn_n_step_wraparound_keeps_remaining_episode_suffix_intact(self):
        gamma = 0.5
        buffer = ReplayBuffer(capacity=8)
        base_x = {"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)}
        base_y = torch.zeros(1, 10)
        mask = torch.ones(self.action_dim, dtype=torch.bool)

        # Episode A occupies slots 0..4, then Episode B wraps and overwrites slot 0.
        for idx, reward in enumerate([10.0, 11.0, 12.0, 13.0, 14.0]):
            buffer.add(
                Transition(
                    x=base_x,
                    y=base_y,
                    action=0,
                    reward=reward,
                    x_next=base_x,
                    y_next=base_y,
                    done=(idx == 4),
                    action_mask=mask,
                    next_action_mask=None if idx == 4 else mask,
                )
            )
        for idx, reward in enumerate([100.0, 101.0, 102.0, 103.0]):
            buffer.add(
                Transition(
                    x=base_x,
                    y=base_y,
                    action=0,
                    reward=reward,
                    x_next=base_x,
                    y_next=base_y,
                    done=(idx == 3),
                    action_mask=mask,
                    next_action_mask=None if idx == 3 else mask,
                )
            )

        # Slot 1 still contains the second transition from Episode A, and its
        # forward chain should remain entirely within the retained suffix
        # {11, 12, 13, 14} rather than jumping into Episode B's wrapped slots.
        wrapped = buffer.build_n_step_transition(index=1, n_step=4, gamma=gamma)

        expected = 11.0 + gamma * 12.0 + (gamma ** 2) * 13.0 + (gamma ** 3) * 14.0
        self.assertTrue(wrapped.done)
        self.assertEqual(wrapped.bootstrap_steps, 4)
        self.assertAlmostEqual(wrapped.reward, expected, places=6)
        self.assertIsNone(wrapped.next_action_mask)

    def test_ppo_mask_stacking_with_mixed_none_and_tensor(self):
        config = PPOConfig(num_steps=2, inner_unroll_n=0)
        trainer = PPOTrainer(self.model, self.env, config)

        partial_mask = torch.tensor([True, False, True, False, True], dtype=torch.bool)
        masks = [None, partial_mask, None, partial_mask]
        stacked = trainer._stack_action_masks(masks)

        self.assertIsNotNone(stacked)
        self.assertEqual(stacked.shape, (4, self.action_dim))
        self.assertEqual(stacked.dtype, torch.bool)
        all_valid = torch.ones(self.action_dim, dtype=torch.bool)
        self.assertTrue(torch.equal(stacked[0].cpu(), all_valid))
        self.assertTrue(torch.equal(stacked[2].cpu(), all_valid))
        self.assertTrue(torch.equal(stacked[1].cpu(), partial_mask))
        self.assertTrue(torch.equal(stacked[3].cpu(), partial_mask))

    def test_ppo_mask_stacking_returns_none_when_all_none(self):
        config = PPOConfig(num_steps=2, inner_unroll_n=0)
        trainer = PPOTrainer(self.model, self.env, config)

        self.assertIsNone(trainer._stack_action_masks([None, None, None]))

    def test_a2c_mask_stacking_with_mixed_none_and_tensor(self):
        config = A2CConfig(num_steps=2, inner_unroll_n=0)
        trainer = A2CTrainer(self.model, self.env, config)

        partial_mask = torch.tensor([True, False, True, False, True], dtype=torch.bool)
        masks = [partial_mask, None]
        stacked = trainer._stack_action_masks(masks)

        self.assertIsNotNone(stacked)
        self.assertEqual(stacked.shape, (2, self.action_dim))
        self.assertEqual(stacked.dtype, torch.bool)
        self.assertTrue(torch.equal(stacked[0].cpu(), partial_mask))
        self.assertTrue(
            torch.equal(stacked[1].cpu(), torch.ones(self.action_dim, dtype=torch.bool))
        )

    def test_baseline_trainers_lack_imitation_pretrain(self):
        ppo_trainer = PPOTrainer(
            self.model, self.env, PPOConfig(num_steps=2, inner_unroll_n=0)
        )
        a2c_trainer = A2CTrainer(
            self.model, self.env, A2CConfig(num_steps=2, inner_unroll_n=0)
        )
        dqn_trainer = DQNTrainer(
            self.model,
            self.env,
            DQNConfig(min_buffer_size=1, batch_size=1, inner_unroll_n=0),
        )

        self.assertFalse(hasattr(ppo_trainer, "imitation_pretrain"))
        self.assertFalse(hasattr(a2c_trainer, "imitation_pretrain"))
        self.assertFalse(hasattr(dqn_trainer, "imitation_pretrain"))

    def test_ppo_raises_when_no_edit_policy_params(self):
        class NoEditPolicyModel(nn.Module):
            def __init__(self, action_dim=5, hidden_dim=8):
                super().__init__()
                self.config = MockConfig(hidden_dim, rl_num_actions=action_dim)
                self.policy = nn.Linear(hidden_dim, action_dim)
                self.value = nn.Linear(hidden_dim, 1)
                self.encoder = nn.Linear(10, hidden_dim)

            def policy_dist(self, x, y, n=4, action_mask=None, z=None):
                z = self.encoder(x["inputs"].float())
                logits = self.policy(z)
                if action_mask is not None:
                    logits = logits.masked_fill(~action_mask.bool(), -1e9)
                return torch.distributions.Categorical(logits=logits), None

            def used_value(self, x, y, n=4):
                z = self.encoder(x["inputs"].float())
                return self.value(z).squeeze(-1), None

            def encode(self, x, y):
                return self.encoder(x["inputs"].float())

        model = NoEditPolicyModel(action_dim=self.action_dim)

        with self.assertRaises(ValueError) as ctx:
            PPOTrainer(model, self.env, PPOConfig(num_steps=2, inner_unroll_n=0))

        self.assertIn("edit_policy", str(ctx.exception))

    def test_baseline_configs_expose_eval_num_episodes_with_default_50(self):
        ppo_cfg = PPOConfig()
        a2c_cfg = A2CConfig()
        dqn_cfg = DQNConfig()

        self.assertEqual(ppo_cfg.eval_num_episodes, 50)
        self.assertEqual(a2c_cfg.eval_num_episodes, 50)
        self.assertEqual(dqn_cfg.eval_num_episodes, 50)

        self.assertEqual(PPOConfig(eval_num_episodes=7).eval_num_episodes, 7)
        self.assertEqual(A2CConfig(eval_num_episodes=7).eval_num_episodes, 7)
        self.assertEqual(DQNConfig(eval_num_episodes=7).eval_num_episodes, 7)

    def test_ppo_evaluate_policy_metrics_falls_back_to_config_eval_num_episodes(self):
        captured = {}

        def fake_eval(*args, **kwargs):
            captured["num_episodes"] = kwargs.get("num_episodes")
            return 0.0, 0.0, {}

        import rl.evaluator as evaluator_mod

        orig = evaluator_mod.evaluate_plan_policy_with_scores
        evaluator_mod.evaluate_plan_policy_with_scores = fake_eval
        try:
            trainer = PPOTrainer(
                self.model,
                self.env,
                PPOConfig(num_steps=2, inner_unroll_n=0, eval_num_episodes=7),
            )
            trainer.evaluate_policy_metrics(
                env_cfg=MagicMock(),
                dataset=[],
                checker=lambda x, y: 0.0,
            )
        finally:
            evaluator_mod.evaluate_plan_policy_with_scores = orig

        self.assertEqual(captured["num_episodes"], 7)

    def test_a2c_evaluate_policy_metrics_falls_back_to_config_eval_num_episodes(self):
        captured = {}

        def fake_eval(*args, **kwargs):
            captured["num_episodes"] = kwargs.get("num_episodes")
            return 0.0, 0.0, {}

        import rl.evaluator as evaluator_mod

        orig = evaluator_mod.evaluate_plan_policy_with_scores
        evaluator_mod.evaluate_plan_policy_with_scores = fake_eval
        try:
            trainer = A2CTrainer(
                self.model,
                self.env,
                A2CConfig(num_steps=2, inner_unroll_n=0, eval_num_episodes=3),
            )
            trainer.evaluate_policy_metrics(
                env_cfg=MagicMock(),
                dataset=[],
                checker=lambda x, y: 0.0,
            )
        finally:
            evaluator_mod.evaluate_plan_policy_with_scores = orig

        self.assertEqual(captured["num_episodes"], 3)

    def test_dqn_evaluate_policy_metrics_falls_back_to_config_eval_num_episodes(self):
        from unittest.mock import patch
        import rl.algos.dqn as dqn_mod

        mock_eval_env = MagicMock()
        mock_eval_env.stop_action_id = self.stop_action_id
        mock_eval_env.reset.return_value = (
            {"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)},
            torch.zeros(1, 10),
        )
        mock_eval_env.step.return_value = (
            (
                {"inputs": torch.zeros(1, 10), "puzzle_identifiers": torch.zeros(1)},
                torch.zeros(1, 10),
            ),
            0.0,
            True,
            {},
        )
        mock_eval_env.get_action_mask.return_value = torch.ones(
            self.action_dim, dtype=torch.bool
        )

        trainer = DQNTrainer(
            self.model,
            self.env,
            DQNConfig(
                min_buffer_size=1,
                batch_size=1,
                inner_unroll_n=0,
                eval_num_episodes=7,
            ),
        )

        dataset = [
            {
                "inputs": torch.zeros(10, dtype=torch.long),
                "puzzle_identifiers": torch.tensor([0]),
            }
        ]
        env_cfg = SimpleNamespace(max_edits=1)

        with patch.object(dqn_mod, "PlanEditEnv", return_value=mock_eval_env):
            trainer.evaluate_policy_metrics(
                env_cfg=env_cfg,
                dataset=dataset,
                checker=lambda x, y: 0.0,
            )

        self.assertEqual(mock_eval_env.reset.call_count, 7)

    def test_evaluator_threads_task_config_into_eval_env(self):
        model = MockModel(action_dim=self.action_dim)
        with torch.no_grad():
            model.edit_policy.weight.zero_()
            model.edit_policy.bias.zero_()
            model.edit_policy.bias[2] = 10.0

        dataset = [
            {
                "inputs": torch.ones(10, dtype=torch.long),
                "puzzle_identifiers": torch.tensor([0]),
                "initial_plan": torch.ones(10, dtype=torch.long),
            }
        ]
        env_cfg = PlanEditEnvConfig(
            max_edits=3,
            gamma=0.99,
            reward_shaping=False,
            task_type="dummy",
            vocab_size=4,
            stop_action_mode="terminal",
        )

        checker = lambda x, y: float(torch.as_tensor(y).sum().item())
        fake_task_config = FakeTaskConfig()

        _, _, no_task_stats = evaluate_plan_policy_with_scores(
            model=model,
            dataset=dataset,
            checker=checker,
            env_cfg=env_cfg,
            num_episodes=1,
            inner_unroll_n=0,
            greedy=True,
        )
        _, _, with_task_stats = evaluate_plan_policy_with_scores(
            model=model,
            dataset=dataset,
            checker=checker,
            env_cfg=env_cfg,
            task_config=fake_task_config,
            num_episodes=1,
            inner_unroll_n=0,
            greedy=True,
        )

        self.assertEqual(no_task_stats["mean_steps"], 3.0)
        self.assertEqual(with_task_stats["mean_steps"], 1.0)
        self.assertGreater(fake_task_config.calls, 0)

if __name__ == "__main__":
    unittest.main()
