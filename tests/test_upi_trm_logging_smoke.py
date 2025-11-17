import torch

from upi_trm_train import DummyPuzzleDataset, dummy_checker
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
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


def test_logging_and_eval_hooks_run():
    dataset = DummyPuzzleDataset(num_instances=4, seq_len=8, vocab_size=16)
    env_cfg = PlanEditEnvConfig(max_edits=3, gamma=0.9, reward_shaping=True, vocab_size=dataset.vocab_size)
    env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg)
    env.set_stop_action_id(stop_id=_num_actions(dataset.seq_len, dataset.vocab_size) - 1)

    rl_cfg = RLConfig(
        batch_size=2,
        num_train_steps=3,
        rollout_episodes_per_step=1,
        max_edits=3,
        log_interval=1,
        eval_interval=2,
        eval_num_episodes=5,
        use_tqdm=False,
    )

    model_cfg = _tiny_trm_cfg(
        seq_len=dataset.seq_len,
        vocab_size=dataset.vocab_size,
        num_identifiers=dataset.num_identifiers,
        batch_size=rl_cfg.batch_size,
    )
    model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
    trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=torch.device("cpu"))

    for _ in range(rl_cfg.num_train_steps):
        metrics = trainer.train_step()
        assert "loss_value" in metrics and "loss_policy" in metrics

    success_rate = trainer.evaluate_policy_success_rate(env_cfg=env_cfg, dataset=dataset, checker=dummy_checker)
    assert isinstance(success_rate, float)
    assert 0.0 <= success_rate <= 1.0

