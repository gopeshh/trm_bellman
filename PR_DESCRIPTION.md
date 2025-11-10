# Scaffold RL workspace (tests & template)

## Summary

This PR establishes the foundation for RL-based meta-MDP training by creating:
1. **CI/CD infrastructure** with pytest, pre-commit hooks (black, isort, flake8), and GitHub Actions workflow
2. **RL package skeleton** with comprehensive stubs for meta-MDP, policy optimization, value estimation, and rollout collection

**Why**: Before implementing RL training logic, we need proper testing infrastructure and a well-defined package structure to ensure code quality and maintainability throughout development.

**What**: This adds safety railings (tests, linters, PR templates) and creates the `rl/` package with 7 core modules containing class/function stubs with comprehensive docstrings.

## Design

### Key Files Added

**CI/CD Infrastructure (Step 0)**:
- `.github/pull_request_template.md` - Structured PR template
- `.github/CONTRIBUTING.md` - Branch naming, commit style, issue labels, branch protection
- `.github/SETUP.md` - Instructions for repository configuration
- `.github/workflows/ci.yml` - Automated testing workflow
- `.pre-commit-config.yaml` - Pre-commit hooks (black, isort, flake8)
- `pyproject.toml` - Tool configurations
- `pytest.ini` - Pytest configuration
- `tests/test_bootstrap.py` - Bootstrap tests (2 passing, 1 skipped)
- `requirements.txt` - Added dev dependencies (pytest, pre-commit, black, isort, flake8)

**RL Package Skeleton (Step 1)**:
- `rl/__init__.py` - Package initialization
- `rl/meta_mdp.py` - `MetaMDP` class with `reset()`, `apply_edit()`, `reward()`, `step()`, `is_terminal()`
- `rl/policy_head.py` - `PolicyHead` and `DiscretePolicyHead` for edit policy πφ
- `rl/value_head.py` - `ValueHead` with EMA target network for Vψ
- `rl/rollout.py` - `RolloutBuffer` and `Rollout` for K-step trajectory collection
- `rl/advantages.py` - Advantage estimation with GAE and centering
- `rl/losses.py` - `RLLoss` with PPO, TRPO, value BR loss, entropy regularization
- `rl/contraction.py` - Spectral norm utilities and Lipschitz monitoring

### API Design

**MetaMDP**: Treats model's internal reasoning as sequential decision problem
```python
mdp = MetaMDP(state_dim=256, action_dim=128)
x, state = mdp.reset(input_tensor)
next_state, reward, done, info = mdp.step(state, edit_action)
```

**Policy & Value Heads**: Neural network stubs for RL training
```python
policy = PolicyHead(state_dim=256, action_dim=128)
value = ValueHead(state_dim=256, ema_decay=0.995)
action, log_prob = policy.sample(x, state)
value_estimate = value.forward(x, state)
```

**Rollout Collection**: K-step trajectories with bootstrapping
```python
rollout = Rollout(mdp, policy, value, K=16, gamma=0.99)
buffer = rollout.collect(input_tensor, num_rollouts=10)
```

## Risk & Mitigation

### Risks
1. **Stub-only implementation**: All methods use `pass` - no actual functionality yet
2. **Type annotations simplified**: Using `Any` for torch.Tensor to avoid import dependencies during bootstrap
3. **No integration tests**: Only bootstrap tests verify imports work

### Mitigation
- ✅ Pre-commit hooks enforce code quality from the start
- ✅ CI workflow will catch issues early when implementations are added
- ✅ Comprehensive docstrings document expected behavior for future implementation
- ✅ Clear separation between modules makes testing easier when functionality is added
- 📋 Next PRs will implement actual functionality module-by-module with corresponding tests

## Test Plan

### Commands
```bash
# Verify pytest passes
python -m pytest -q tests/
# Expected: 2 passed, 1 skipped

# Verify imports work
python - <<'PY'
import rl
import rl.meta_mdp
import rl.policy_head
import rl.value_head
print('ok')
PY
# Expected: ok

# Verify pre-commit passes
pre-commit run --all-files
# Expected: All checks pass (black, isort, flake8, etc.)

# Verify git status
git log --oneline -2
# Expected: Shows both commits
```

### Test Results
```
============================= test session starts ==============================
platform darwin -- Python 3.11.14, pytest-9.0.0, pluggy-1.6.0
rootdir: /Users/buiksat/trm_bellman
configfile: pytest.ini
plugins: cov-7.0.0
collected 3 items

tests/test_bootstrap.py ..s                                              [100%]

========================= 2 passed, 1 skipped in 0.00s =========================
```

All pre-commit hooks passing ✅

## Metrics/Charts

**Code Coverage**: N/A (stubs only)

**Code Quality**:
- ✅ 100% black formatted
- ✅ 100% isort compliant
- ✅ 0 flake8 violations
- ✅ All imports resolve successfully

**Package Structure**:
- 8 RL modules created
- ~1,000+ lines of docstrings
- 50+ function/method stubs
- 12 class definitions

## Scope Creep

**Explicitly out-of-scope for this PR**:
- ❌ Actual implementation of RL algorithms (PPO, value functions, etc.)
- ❌ Integration with existing TRM models
- ❌ Training loops or optimization logic
- ❌ Real reward functions or MDP dynamics
- ❌ Performance benchmarks or metrics
- ❌ Documentation beyond docstrings (README updates, tutorials)
- ❌ Unit tests for individual RL components

**Next PRs will handle**:
- Implementing meta-MDP dynamics
- Policy and value network implementations
- Training loop integration
- Task-specific reward functions
- Comprehensive unit tests
