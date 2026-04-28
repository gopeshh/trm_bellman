"""Tests for CleanRL DQN n-step buffer correctness."""

import numpy as np
import pytest


def test_nstep_buffer_1step_passthrough():
    """With n=1, transitions pass through unchanged."""
    from rl.cleanrl.dqn_trm import NStepBuffer, ReplayBuffer, Transition

    replay = ReplayBuffer(100)
    buf = NStepBuffer(n=1, gamma=0.99, replay=replay)

    buf.push(np.array([1.0]), 0, 1.0, np.array([2.0]), False, episode_done=False)
    assert len(replay) == 1
    t = replay.buffer[0]
    assert t.reward == 1.0
    assert t.n_steps == 1
    assert t.done is False


def test_nstep_buffer_accumulates_rewards():
    """With n=3, reward is gamma-discounted sum of 3 steps."""
    from rl.cleanrl.dqn_trm import NStepBuffer, ReplayBuffer

    replay = ReplayBuffer(100)
    gamma = 0.99
    buf = NStepBuffer(n=3, gamma=gamma, replay=replay)

    buf.push(np.array([1.0]), 0, 1.0, np.array([2.0]), False)
    buf.push(np.array([2.0]), 1, 2.0, np.array([3.0]), False)
    buf.push(np.array([3.0]), 2, 3.0, np.array([4.0]), False)
    assert len(replay) == 1
    t = replay.buffer[0]
    expected = 1.0 + gamma * 2.0 + gamma**2 * 3.0
    assert abs(t.reward - expected) < 1e-6
    assert t.n_steps == 3
    assert np.allclose(t.next_obs, [4.0])
    assert t.done is False


def test_nstep_buffer_episode_boundary_flushes_all():
    """Episode end flushes all pending transitions with correct walk lengths."""
    from rl.cleanrl.dqn_trm import NStepBuffer, ReplayBuffer

    replay = ReplayBuffer(100)
    gamma = 0.99
    buf = NStepBuffer(n=5, gamma=gamma, replay=replay)

    buf.push(np.array([1.0]), 0, 1.0, np.array([2.0]), False)
    buf.push(np.array([2.0]), 1, 2.0, np.array([3.0]), True, episode_done=True)
    assert len(replay) == 2

    t0 = replay.buffer[0]
    assert t0.n_steps == 2
    assert t0.done is True
    expected_r0 = 1.0 + gamma * 2.0
    assert abs(t0.reward - expected_r0) < 1e-6

    t1 = replay.buffer[1]
    assert t1.n_steps == 1
    assert t1.done is True
    assert t1.reward == 2.0


def test_nstep_buffer_truncation_flushes():
    """Truncated episodes (episode_done=True, done=False) flush properly."""
    from rl.cleanrl.dqn_trm import NStepBuffer, ReplayBuffer

    replay = ReplayBuffer(100)
    buf = NStepBuffer(n=5, gamma=0.99, replay=replay)

    buf.push(np.array([1.0]), 0, 1.0, np.array([2.0]), False)
    buf.push(np.array([2.0]), 1, 2.0, np.array([3.0]), False, episode_done=True)
    assert len(replay) == 2
    assert replay.buffer[0].done is False
    assert replay.buffer[0].n_steps == 2
    assert replay.buffer[1].done is False
    assert replay.buffer[1].n_steps == 1


def test_nstep_buffer_no_cross_episode_leak():
    """Rewards from episode N must not leak into episode N+1."""
    from rl.cleanrl.dqn_trm import NStepBuffer, ReplayBuffer

    replay = ReplayBuffer(100)
    gamma = 0.99
    buf = NStepBuffer(n=3, gamma=gamma, replay=replay)

    buf.push(np.array([1.0]), 0, 10.0, np.array([2.0]), True, episode_done=True)
    buf.push(np.array([10.0]), 0, 1.0, np.array([11.0]), False)
    buf.push(np.array([11.0]), 1, 1.0, np.array([12.0]), False)
    buf.push(np.array([12.0]), 2, 1.0, np.array([13.0]), False)

    assert len(replay) == 2
    t_ep1 = replay.buffer[0]
    assert t_ep1.reward == 10.0
    assert t_ep1.done is True

    t_ep2 = replay.buffer[1]
    expected = 1.0 + gamma * 1.0 + gamma**2 * 1.0
    assert abs(t_ep2.reward - expected) < 1e-6
    assert t_ep2.done is False
    assert t_ep2.n_steps == 3


def test_transition_n_steps_default():
    """Transition.n_steps defaults to 1 for 1-step DQN."""
    from rl.cleanrl.dqn_trm import Transition

    t = Transition(obs=np.array([0]), action=0, reward=1.0, next_obs=np.array([1]), done=False)
    assert t.n_steps == 1


def test_nstep_gamma09_hand_constructed():
    """Hand-constructed reference case with gamma=0.9 for exact verification.

    Sequence: s0 --(a=0,r=1)--> s1 --(a=1,r=2)--> s2 --(a=2,r=3)--> s3

    With n=3, gamma=0.9:
      Transition from s0: reward = 1 + 0.9*2 + 0.81*3 = 5.23, next=s3, done=F, n_steps=3
    With n=2, gamma=0.9:
      Transition from s0: reward = 1 + 0.9*2 = 2.8, next=s2, done=F, n_steps=2
      Transition from s1: reward = 2 + 0.9*3 = 4.7, next=s3, done=F, n_steps=2
    """
    from rl.cleanrl.dqn_trm import NStepBuffer, ReplayBuffer

    gamma = 0.9

    # n=3 case
    replay3 = ReplayBuffer(100)
    buf3 = NStepBuffer(n=3, gamma=gamma, replay=replay3)
    buf3.push(np.array([0.0]), 0, 1.0, np.array([1.0]), False)
    buf3.push(np.array([1.0]), 1, 2.0, np.array([2.0]), False)
    buf3.push(np.array([2.0]), 2, 3.0, np.array([3.0]), False)
    assert len(replay3) == 1
    t = replay3.buffer[0]
    expected = 1.0 + 0.9 * 2.0 + 0.81 * 3.0  # 5.23
    assert abs(t.reward - expected) < 1e-9, f"n=3: got {t.reward}, expected {expected}"
    assert t.n_steps == 3
    assert np.allclose(t.next_obs, [3.0])

    # n=2 case
    replay2 = ReplayBuffer(100)
    buf2 = NStepBuffer(n=2, gamma=gamma, replay=replay2)
    buf2.push(np.array([0.0]), 0, 1.0, np.array([1.0]), False)
    buf2.push(np.array([1.0]), 1, 2.0, np.array([2.0]), False)
    assert len(replay2) == 1
    t0 = replay2.buffer[0]
    expected0 = 1.0 + 0.9 * 2.0  # 2.8
    assert abs(t0.reward - expected0) < 1e-9, f"n=2 t0: got {t0.reward}, expected {expected0}"
    assert t0.n_steps == 2
    buf2.push(np.array([2.0]), 2, 3.0, np.array([3.0]), False)
    assert len(replay2) == 2
    t1 = replay2.buffer[1]
    expected1 = 2.0 + 0.9 * 3.0  # 4.7
    assert abs(t1.reward - expected1) < 1e-9, f"n=2 t1: got {t1.reward}, expected {expected1}"
    assert t1.n_steps == 2
