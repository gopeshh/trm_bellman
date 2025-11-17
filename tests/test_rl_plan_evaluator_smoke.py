import torch

from evaluators.rl_plan_evaluator import evaluate_plan_policy
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.envs.plan_edit_env import PlanEditEnvConfig
from upi_trm_train import DummyPuzzleDataset, dummy_checker


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
        rl_num_actions=4,
    )


def test_evaluate_plan_policy_smoke():
    torch.manual_seed(0)

    dataset = DummyPuzzleDataset(num_instances=8, seq_len=12, vocab_size=16)
    env_cfg = PlanEditEnvConfig(max_edits=4, gamma=0.99, reward_shaping=True)

    batch_size = 4
    cfg = _tiny_trm_cfg(
        seq_len=dataset.seq_len,
        vocab_size=dataset.vocab_size,
        num_identifiers=dataset.num_identifiers,
        batch_size=batch_size,
    )
    model = TinyRecursiveReasoningModel_ACTV1(cfg)

    score = evaluate_plan_policy(
        model=model,
        dataset=dataset,
        checker=dummy_checker,
        env_cfg=env_cfg,
        num_episodes=10,
        inner_unroll_n=2,
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

