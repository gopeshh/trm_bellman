import math

import torch

from upi_trm_train import DummyPuzzleDataset, dummy_checker
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.task_config import DummyTaskConfig
from rl.upi_trm_trainer import UPITrmTrainer


def _num_actions(seq_len: int, vocab_size: int) -> int:
    return seq_len * vocab_size + 1


def _tiny_trm_cfg(seq_len: int, vocab_size: int, num_identifiers: int, batch_size: int):
    return dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=max(num_identifiers, batch_size),
        vocab_size=vocab_size,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,
        L_layers=1,
        hidden_size=32,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=0,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=False,
        rl_enable_policy_head=True,
        rl_num_actions=_num_actions(seq_len, vocab_size),
    )


def test_upi_trm_trainer_smoke():
    dataset = DummyPuzzleDataset(num_instances=8, seq_len=12, vocab_size=16)
    env_cfg = PlanEditEnvConfig(max_edits=4, gamma=0.99, reward_shaping=True, vocab_size=dataset.vocab_size)
    env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg)
    env.set_stop_action_id(stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1)

    rl_cfg = RLConfig(
        batch_size=4,
        num_train_steps=5,
        rollout_episodes_per_step=1,
        max_edits=4,
    )

    model_cfg = _tiny_trm_cfg(
        seq_len=dataset.seq_len,
        vocab_size=dataset.vocab_size,
        num_identifiers=dataset.num_identifiers,
        batch_size=rl_cfg.batch_size,
    )
    model = TinyRecursiveReasoningModel_ACTV1(model_cfg)

    trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))

    for _ in range(3):
        metrics = trainer.train_step()
        assert "loss_value" in metrics and "loss_policy" in metrics
        assert math.isfinite(metrics["loss_value"])
        assert math.isfinite(metrics["loss_policy"])


class TrackingTaskConfig(DummyTaskConfig):
    def __init__(self):
        super().__init__()
        self.batch_called = False
        self.last_current_state_shape = None

    def compute_batch_action_mask(self, inputs, vocab_size, stop_action_id, current_state=None):
        if current_state is None:
            raise AssertionError("current_state should be provided for batch masking")
        self.batch_called = True
        if torch.is_tensor(current_state):
            self.last_current_state_shape = tuple(current_state.shape)
        return super().compute_batch_action_mask(inputs, vocab_size, stop_action_id, current_state=current_state)


def test_policy_update_uses_task_config_batch_mask():
    dataset = DummyPuzzleDataset(num_instances=6, seq_len=8, vocab_size=12)
    env_cfg = PlanEditEnvConfig(max_edits=3, gamma=0.99, reward_shaping=True, vocab_size=dataset.vocab_size)
    task_config = TrackingTaskConfig()
    env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg, task_config=task_config)
    env.set_stop_action_id(stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1)

    rl_cfg = RLConfig(
        batch_size=2,
        num_train_steps=4,
        rollout_episodes_per_step=2,
        max_edits=3,
    )
    model_cfg = _tiny_trm_cfg(
        seq_len=dataset.seq_len,
        vocab_size=dataset.vocab_size,
        num_identifiers=dataset.num_identifiers,
        batch_size=rl_cfg.batch_size,
    )
    model = TinyRecursiveReasoningModel_ACTV1(model_cfg)

    trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))
    for _ in range(3):
        trainer.train_step()

    assert task_config.batch_called
    assert task_config.last_current_state_shape is not None

