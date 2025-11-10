"""
Logging utilities for RL training.

Provides stdout logging and optional Weights & Biases integration.
"""
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class RLLogger:
    """Logger for RL training metrics with stdout and optional W&B support."""

    def __init__(
        self,
        use_wandb: bool = False,
        log_interval: int = 10,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize the RL logger.

        Args:
            use_wandb: Whether to use Weights & Biases for logging
            log_interval: How often to log (in steps)
            project_name: W&B project name (if use_wandb=True)
            run_name: W&B run name (if use_wandb=True)
        """
        self.use_wandb = use_wandb
        self.log_interval = log_interval
        self.step = 0

        if self.use_wandb:
            try:
                import wandb

                wandb.init(project=project_name, name=run_name)
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, disabling W&B logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to stdout and optionally to W&B.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step counter (uses internal counter if None)
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1

        # Only log at specified intervals
        if self.step % self.log_interval != 0:
            return

        # Convert tensors to Python scalars
        clean_metrics = {}
        for key, value in metrics.items():
            if torch is not None and isinstance(value, torch.Tensor):
                clean_metrics[key] = value.item()
            else:
                clean_metrics[key] = value

        # Log to stdout
        metric_str = " | ".join(
            [
                f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in clean_metrics.items()
            ]
        )
        print(f"[Step {self.step}] {metric_str}")

        # Log to W&B if enabled
        if self.use_wandb and self.wandb is not None:
            self.wandb.log(clean_metrics, step=self.step)

    def finish(self) -> None:
        """Finish logging and cleanup."""
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()


def compute_value_residual(values: "torch.Tensor", returns: "torch.Tensor") -> "torch.Tensor":
    """
    Compute max absolute value residual (Bellman error) across batch.

    Args:
        values: Value predictions V_ψ(z_n, x), shape [B]
        returns: Target returns G^(K), shape [B]

    Returns:
        Scalar tensor: max|V_ψ - G^(K)| over batch
    """
    if torch is None:
        raise ImportError("torch is required for compute_value_residual")
    return (values - returns).abs().max()


def compute_policy_kl(
    log_probs_new: "torch.Tensor", log_probs_old: "torch.Tensor"
) -> "torch.Tensor":
    """
    Compute KL divergence KL(π_old || π_new) = E[log π_old - log π_new].

    Args:
        log_probs_new: Log probabilities from current policy, shape [B]
        log_probs_old: Log probabilities from old policy, shape [B]

    Returns:
        Scalar tensor: mean KL divergence
    """
    if torch is None:
        raise ImportError("torch is required for compute_policy_kl")
    return (log_probs_old - log_probs_new).mean()


def compute_policy_entropy(
    log_probs_pos: "torch.Tensor", log_probs_val: "torch.Tensor"
) -> "torch.Tensor":
    """
    Compute policy entropy H(π) = -E[log π] for factorized policy.

    For a factorized policy π(a) = π(pos) * π(val|pos), the entropy is:
    H(π) = H(pos) + E_pos[H(val|pos)]

    For simplicity, we approximate: H(π) ≈ -(log_prob_pos + log_prob_val)

    Args:
        log_probs_pos: Log probabilities over positions, shape [B, num_cells]
        log_probs_val: Log probabilities over values, shape [B, num_vals]

    Returns:
        Scalar tensor: mean entropy estimate
    """
    if torch is None:
        raise ImportError("torch is required for compute_policy_entropy")
    # Entropy for position distribution
    probs_pos = log_probs_pos.exp()
    entropy_pos = -(probs_pos * log_probs_pos).sum(dim=-1)

    # Entropy for value distribution
    probs_val = log_probs_val.exp()
    entropy_val = -(probs_val * log_probs_val).sum(dim=-1)

    # Total entropy (approximation)
    return (entropy_pos + entropy_val).mean()


def compute_ppo_clip_fraction(
    log_probs_new: "torch.Tensor",
    log_probs_old: "torch.Tensor",
    epsilon: float = 0.2,
) -> "torch.Tensor":
    """
    Compute fraction of policy updates that were clipped by PPO.

    Args:
        log_probs_new: Log probabilities from current policy, shape [B]
        log_probs_old: Log probabilities from old policy, shape [B]
        epsilon: PPO clipping parameter

    Returns:
        Scalar tensor: fraction of samples where |ratio - 1| > epsilon
    """
    if torch is None:
        raise ImportError("torch is required for compute_ppo_clip_fraction")
    ratio = (log_probs_new - log_probs_old).exp()
    clipped = ((ratio - 1.0).abs() > epsilon).float()
    return clipped.mean()


def compute_spectral_product(module: "torch.nn.Module") -> float:
    """
    Compute product of spectral norms across layers with spectral_norm.

    Args:
        module: PyTorch module with spectral_norm applied

    Returns:
        Product of spectral norms
    """
    if torch is None:
        raise ImportError("torch is required for compute_spectral_product")

    product = 1.0
    for name, param in module.named_parameters():
        # Check if this parameter has spectral norm applied
        # The spectral norm parametrization stores the singular value in a buffer
        if "weight_u" in name or "weight_v" in name:
            continue  # Skip the u/v vectors used by spectral norm

        # Look for the corresponding spectral norm buffer
        if name.endswith(".weight"):
            base_name = name[:-7]  # Remove ".weight"
            # Try to find the spectral norm buffer
            sigma_name = f"{base_name}.weight_sigma"
            if hasattr(module, sigma_name.replace(".", "_")):
                # Get the spectral norm value
                sigma = getattr(module, sigma_name.replace(".", "_"))
                if torch.is_tensor(sigma):
                    product *= sigma.item()

    return product


def compute_score_delta(
    scores_before: "torch.Tensor", scores_after: "torch.Tensor"
) -> "torch.Tensor":
    """
    Compute mean score improvement delta.

    Args:
        scores_before: Task scores before edits, shape [B]
        scores_after: Task scores after edits, shape [B]

    Returns:
        Scalar tensor: mean score improvement
    """
    if torch is None:
        raise ImportError("torch is required for compute_score_delta")
    return (scores_after - scores_before).mean()
