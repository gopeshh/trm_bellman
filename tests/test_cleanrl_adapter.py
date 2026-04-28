"""Tests for CleanRL adapter components: GymPlanEditEnv, TRM adapters, bundle builder."""

import torch
import pytest


def test_gym_plan_edit_env_step_returns_five_tuple():
    """GymPlanEditEnv.step must return (obs, reward, terminated, truncated, info)."""
    from rl.cleanrl.trm_adapter import GymPlanEditEnv, build_sudoku_bundle

    config = {"dataset_path": "data/sudoku-4x4-trivial", "max_edits": 4, "batch_size": 4}
    bundle = build_sudoku_bundle(config)
    env = GymPlanEditEnv(
        dataset=bundle.dataset, checker_fn=bundle.checker_fn,
        env_cfg=bundle.env_cfg, task_config=bundle.task_config, seed=0,
    )
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert "inputs" in obs and "plan" in obs and "action_mask" in obs

    result = env.step(0)
    assert len(result) == 5, f"step() must return 5-tuple, got {len(result)}"
    obs2, reward, terminated, truncated, info2 = result
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_gym_plan_edit_env_budget_is_truncation():
    """Budget exhaustion must set truncated=True, terminated=False."""
    from rl.cleanrl.trm_adapter import GymPlanEditEnv, build_sudoku_bundle

    config = {"dataset_path": "data/sudoku-4x4-trivial", "max_edits": 2, "batch_size": 4}
    bundle = build_sudoku_bundle(config)
    env = GymPlanEditEnv(
        dataset=bundle.dataset, checker_fn=bundle.checker_fn,
        env_cfg=bundle.env_cfg, task_config=bundle.task_config, seed=0,
    )
    obs, _ = env.reset()
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert truncated or terminated, "Episode should have ended"
    if info.get("terminated_by_budget"):
        assert truncated and not terminated, "Budget exhaustion must be truncated, not terminated"


def test_build_sudoku_bundle_checker_fields():
    """build_sudoku_bundle must forward checker-selection fields to RLConfig."""
    from rl.cleanrl.trm_adapter import build_sudoku_bundle

    config = {
        "dataset_path": "data/sudoku-4x4-trivial",
        "batch_size": 4,
        "use_feasibility_checker": True,
        "feasibility_violation_weight": 3.0,
        "feasibility_zerocand_weight": 7.0,
    }
    bundle = build_sudoku_bundle(config)
    assert bundle.checker_kind == "feasibility"


def test_build_sudoku_bundle_solved_threshold_null():
    """solved_threshold: null in YAML must not crash."""
    from rl.cleanrl.trm_adapter import build_sudoku_bundle

    config = {
        "dataset_path": "data/sudoku-4x4-trivial",
        "batch_size": 4,
        "solved_threshold": None,
    }
    bundle = build_sudoku_bundle(config)
    assert bundle.env_cfg.solved_threshold is None


def test_build_trm_cfg_uses_rlconfig_defaults():
    """_build_trm_cfg must inherit RLConfig defaults, not hardcode zeros."""
    from rl.cleanrl.trm_adapter import _build_trm_cfg, build_sudoku_bundle

    config = {"dataset_path": "data/sudoku-4x4-trivial", "batch_size": 4}
    bundle = build_sudoku_bundle(config)
    trm_cfg = _build_trm_cfg(config, bundle)

    from rl.config import RLConfig
    rl_defaults = RLConfig(batch_size=4, max_edits=16, gamma=0.99)
    assert trm_cfg["rl_enable_contraction"] == rl_defaults.enable_contraction
    assert trm_cfg["rl_target_Lz"] == rl_defaults.target_Lz
    assert trm_cfg["rl_latent_ball_radius"] == getattr(rl_defaults, "latent_ball_radius", 10.0)


def test_pool_size_matches_in_house():
    """Default pool_size must be max(batch_size, 8), same as upi_trm_train.py."""
    from rl.cleanrl.trm_adapter import build_sudoku_bundle

    for bs in [4, 16]:
        config = {"dataset_path": "data/sudoku-4x4-trivial", "batch_size": bs}
        bundle = build_sudoku_bundle(config)
        expected_pool = max(bs, 8)
        assert len(bundle.dataset) <= expected_pool, (
            f"batch_size={bs}: dataset has {len(bundle.dataset)} samples, "
            f"expected at most {expected_pool}"
        )


def test_trm_actor_critic_interface():
    """TRMActorCritic must expose get_value, get_policy_logits, get_action_and_value."""
    from rl.cleanrl.trm_adapter import TRMActorCritic, build_sudoku_bundle

    config = {
        "dataset_path": "data/sudoku-4x4-trivial",
        "batch_size": 4,
        "hidden_size": 32,
        "h_cycles": 1, "l_cycles": 1,
    }
    bundle = build_sudoku_bundle(config)
    agent_cfg = dict(config)
    agent_cfg["_bundle"] = bundle
    agent = TRMActorCritic(agent_cfg)

    import numpy as np
    seq_len = bundle.seq_len
    obs = {
        "inputs": torch.zeros(1, seq_len),
        "plan": torch.zeros(1, seq_len),
        "puzzle_identifiers": torch.zeros(1, 1),
        "action_mask": torch.ones(1, bundle.num_actions, dtype=torch.bool),
    }
    with torch.no_grad():
        value = agent.get_value(obs)
        assert value.shape == (1, 1)

        logits = agent.get_policy_logits(obs)
        assert logits.shape == (1, bundle.num_actions)

        action, logprob, entropy, val = agent.get_action_and_value(obs)
        assert action.shape == (1,)


def test_action_mask_blocks_masked_actions():
    """Masked actions must get -inf logits and never be selected by argmax."""
    from rl.cleanrl.trm_adapter import apply_action_mask

    logits = torch.zeros(1, 5)
    mask = torch.tensor([[True, False, True, False, True]])
    masked = apply_action_mask(logits, mask)
    selected = masked.argmax(dim=-1).item()
    assert selected in {0, 2, 4}, f"argmax selected masked action {selected}"
    assert masked[0, 1].item() < -1e6, "Masked action logit should be very negative"
    assert masked[0, 3].item() < -1e6, "Masked action logit should be very negative"


def test_action_mask_prevents_sampling():
    """Sampling from masked logits must never produce a masked action."""
    from rl.cleanrl.trm_adapter import apply_action_mask
    from torch.distributions import Categorical

    logits = torch.zeros(1, 4)
    mask = torch.tensor([[True, True, False, False]])
    masked = apply_action_mask(logits, mask)
    dist = Categorical(logits=masked)
    samples = dist.sample((1000,)).squeeze()
    unique = set(samples.tolist())
    assert unique.issubset({0, 1}), f"Sampled masked actions: {unique - {0, 1}}"
