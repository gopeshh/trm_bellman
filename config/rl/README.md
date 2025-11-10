# RL Configuration

This directory contains configuration files for Reinforcement Learning training.

## Usage

### Enable RL Training

To enable RL training with default parameters:

```bash
python pretrain.py +rl=default
```

Or explicitly:

```bash
python pretrain.py +rl.enabled=true
```

### Disable RL Training

```bash
python pretrain.py +rl.enabled=false
```

### Override Specific Parameters

Override individual RL hyperparameters:

```bash
# Change discount factor
python pretrain.py +rl=default +rl.gamma=0.99

# Change K-step rollout length
python pretrain.py +rl=default +rl.K=5

# Disable spectral normalization
python pretrain.py +rl=default +rl.spectral.enabled=false

# Use TRPO instead of PPO
python pretrain.py +rl=default +rl.use_trpo=true
```

### Multiple Overrides

```bash
python pretrain.py +rl=default \
  +rl.gamma=0.99 \
  +rl.K=5 \
  +rl.alpha_pi=0.5 \
  +rl.spectral.target_prod=0.9
```

## Configuration Parameters

### Core RL Hyperparameters

- `enabled` (bool): Main toggle for RL training
- `gamma` (float): Discount factor for returns (default: 0.985)
- `K` (int): Number of rollout steps for K-step returns (default: 3)
- `n_inner` (int): Number of inner TRM iterations/recurrence depth (default: 6)
- `tau_ema` (float): EMA decay rate for value target network (default: 0.995)

### Loss Weights

- `alpha_br` (float): Weight for Bellman residual (value) loss (default: 1.0)
- `alpha_pi` (float): Weight for policy loss (default: 1.0)
- `alpha_sup` (float): Weight for supervised loss in mixed training (default: 1.0)

### Policy Optimization

- `entropy_beta` (float): Entropy regularization coefficient (default: 0.001)
- `ppo_clip` (float): PPO clipping epsilon, ratio range [1-ε, 1+ε] (default: 0.2)
- `trpo_kl_target` (float): TRPO KL divergence target (default: 0.01)
- `use_trpo` (bool): Use TRPO instead of PPO (default: false)

### Spectral Normalization

Control Lipschitz constant for contraction:

- `spectral.enabled` (bool): Apply spectral norm to z→z path (default: true)
- `spectral.target_prod` (float): Target cumulative spectral norm product (default: 0.95)
- `spectral.weight` (float): Regularizer weight for R_lip loss (default: 1e-4)

### Logging

- `logging.log_Lz` (bool): Log empirical Lipschitz constant L_z (default: true)
- `logging.log_residuals` (bool): Log Bellman residuals for diagnostics (default: true)

## Creating Custom Configurations

Create a new config file `config/rl/custom.yaml`:

```yaml
# @package _global_.rl
defaults:
  - default

# Override specific parameters
gamma: 0.99
K: 5
spectral:
  target_prod: 0.9
```

Then use it:

```bash
python pretrain.py +rl=custom
```

## Integration with Training Script

In your training script:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="cfg_pretrain")
def main(cfg: DictConfig):
    # Check if RL is enabled
    if cfg.get("rl", {}).get("enabled", False):
        print("RL training enabled!")
        print(f"Discount factor: {cfg.rl.gamma}")
        print(f"K-step rollout: {cfg.rl.K}")

        # Initialize RL components
        # ... your RL setup code ...
    else:
        print("Standard supervised training")
        # ... your supervised training code ...

if __name__ == "__main__":
    main()
```
