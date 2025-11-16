from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def _make_small_config():
    return dict(
        batch_size=2,
        seq_len=4,
        puzzle_emb_ndim=8,
        num_puzzle_identifiers=3,
        vocab_size=16,
        H_cycles=2,
        L_cycles=2,
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
        puzzle_emb_len=4,
        no_ACT_continue=True,
        rl_enable_value_head=True,
        rl_enable_contraction=True,
    )


def test_spectral_norm_is_applied_to_inner_and_value_head():
    config = _make_small_config()
    model = TinyRecursiveReasoningModel_ACTV1(config)

    has_sn_inner = False
    for module in model.inner.modules():
        if getattr(module, "weight_u", None) is not None and getattr(module, "weight_v", None) is not None:
            has_sn_inner = True
            break
    assert has_sn_inner, "Expected at least one spectral-norm-wrapped linear in inner model"

    assert model.value_head is not None
    has_sn_value = False
    for module in model.value_head.modules():
        if getattr(module, "weight_u", None) is not None and getattr(module, "weight_v", None) is not None:
            has_sn_value = True
            break
    assert has_sn_value, "Expected spectral norm on at least one linear in value head"

