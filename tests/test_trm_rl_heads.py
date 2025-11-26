import torch
from torch.distributions import Categorical

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def _num_actions(seq_len: int, vocab_size: int) -> int:
    return seq_len * vocab_size + 1


def _tiny_trm_cfg(batch_size: int, seq_len: int, vocab_size: int, num_identifiers: int):
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
        hidden_size=16,
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


def test_trm_rl_heads_used_value_and_policy_dist_shapes():
    torch.manual_seed(0)

    batch_size = 3
    seq_len = 8
    vocab_size = 16
    num_identifiers = 4

    cfg = _tiny_trm_cfg(batch_size, seq_len, vocab_size, num_identifiers)
    model = TinyRecursiveReasoningModel_ACTV1(cfg)

    inputs = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    puzzle_ids = torch.randint(low=0, high=num_identifiers, size=(batch_size,))
    x_batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

    y_batch = torch.zeros_like(inputs)

    values = model.used_value(x_batch, y_batch, n=2)
    assert values.shape == (batch_size,)

    dist = model.policy_dist(x_batch, y_batch, n=2)
    assert isinstance(dist, Categorical)
    assert dist.logits.shape == (batch_size, cfg["rl_num_actions"])

    probs = dist.probs
    assert probs.shape == (batch_size, cfg["rl_num_actions"])
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def test_used_value_changes_with_plan_embedding():
    torch.manual_seed(1)

    batch_size = 2
    seq_len = 6
    vocab_size = 10
    num_identifiers = 3

    cfg = _tiny_trm_cfg(batch_size, seq_len, vocab_size, num_identifiers)
    model = TinyRecursiveReasoningModel_ACTV1(cfg)

    inputs = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    puzzle_ids = torch.randint(low=0, high=num_identifiers, size=(batch_size,))
    x_batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

    y_plan = torch.zeros_like(inputs)
    y_alt = y_plan.clone()
    y_alt[:, 0] = (y_alt[:, 0] + 1) % vocab_size

    values_base = model.used_value(x_batch, y_plan, n=1)
    values_alt = model.used_value(x_batch, y_alt, n=1)

    assert not torch.allclose(values_base, values_alt)

