#!/usr/bin/env python3
"""Sanity check for vectorized action mask implementation."""

import torch
from rl.task_config import SudokuTaskConfig

def assert_mask_properties(mask, vocab_size, stop_action_id, num_positions):
    assert mask.dtype == torch.bool
    assert mask.shape[-1] == stop_action_id + 1
    # STOP always valid
    assert mask[..., stop_action_id].all()
    # PAD/empty tokens masked
    for tok in [0, 1]:
        idx = torch.arange(num_positions) * vocab_size + tok
        idx = idx[idx < stop_action_id]
        assert (~mask[..., idx]).all()

def test_vectorized_mask():
    cfg = SudokuTaskConfig()
    vocab = 6  # 4x4
    num_positions = 16
    stop = num_positions * vocab

    # inputs: given cells at positions 0 and 15
    inputs = torch.ones(num_positions, dtype=torch.long)
    inputs[0] = 2
    inputs[15] = 5

    # current_state: place digit 1 at (0,0)
    current_state = inputs.clone()
    current_state[0] = 2  # token 2 = digit 1

    mask = cfg.compute_action_mask(inputs, vocab, stop, current_state=current_state)
    assert_mask_properties(mask, vocab, stop, num_positions)

    # given cells fully masked
    for pos in [0, 15]:
        start = pos * vocab
        end = start + vocab
        assert not mask[start:end].any()

    # row constraint: token 2 masked at row 0 positions 1,2,3
    for pos in [1, 2, 3]:
        action_idx = pos * vocab + 2
        assert not mask[action_idx]

    # current_state None should match current_state=inputs
    mask_a = cfg.compute_action_mask(inputs, vocab, stop, current_state=None)
    mask_b = cfg.compute_action_mask(inputs, vocab, stop, current_state=inputs)
    assert torch.equal(mask_a, mask_b)

    # batched path
    bmask = cfg.compute_batch_action_mask(
        inputs.unsqueeze(0), vocab, stop, current_state=current_state.unsqueeze(0)
    )
    assert bmask.shape == (1, stop + 1)
    assert torch.equal(bmask[0], mask)

    print("✓ All assertions passed")

if __name__ == "__main__":
    test_vectorized_mask()
    print("OK")
