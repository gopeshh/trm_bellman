"""Minimal RL Training Loop for TRM with Meta-MDP.

This script implements reinforcement learning for training TRM models to edit
their own internal states through policy optimization.
"""

import os
import warnings
from typing import Optional

import hydra
import torch
import torch.nn as nn
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Import existing data loading helpers
from pretrain import PretrainConfig, create_dataloader
from puzzle_dataset import PuzzleDatasetMetadata

# Import RL components
from rl.advantages import center_advantages, compute_advantages
from rl.contraction import LipschitzMonitor, apply_spectral_norm, estimate_Lz
from rl.losses import ppo_policy_loss, value_bellman_residual_loss
from rl.meta_mdp import SudokuMetaMDP
from rl.policy_head import EditPolicy
from rl.rollout import rollout_k
from rl.value_head import ValueHead

warnings.filterwarnings("ignore")


class TRMWrapper(nn.Module):
    """Wrapper around TRM model for RL training.

    Exposes inner_unroll for computing recurrent states.
    """

    def __init__(self, base_model: nn.Module, n_inner: int = 6):
        """Initialize TRM wrapper.

        Args:
            base_model: Base TRM model
            n_inner: Number of inner recurrence steps
        """
        super().__init__()
        self.model = base_model
        self.n_inner = n_inner

    def inner_unroll(
        self, z: torch.Tensor, y: torch.Tensor, x: torch.Tensor, n: Optional[int] = None
    ) -> torch.Tensor:
        """Perform n steps of inner recurrence.

        Args:
            z: Initial internal state (B, z_dim)
            y: Current output (B, seq_len)
            x: Input (B, seq_len)
            n: Number of steps (uses self.n_inner if None)

        Returns:
            Final internal state z_n after n steps
        """
        n = n if n is not None else self.n_inner

        # Simplified inner unroll - adapt based on actual TRM architecture
        # This is a placeholder that should be replaced with actual TRM logic
        z_current = z
        for _ in range(n):
            # z_{t+1} = f_theta(z_t, y, x)
            # In real TRM, this would call the actual recurrence function
            z_current = self.model(z_current, y, x)  # Placeholder

        return z_current

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for supervised loss.

        Args:
            y: Current output (B, seq_len)
            x: Input (B, seq_len)

        Returns:
            Predicted output
        """
        return self.model(y, x)


class RLTrainer:
    """RL Trainer for TRM with Meta-MDP."""

    def __init__(
        self,
        model: TRMWrapper,
        policy: EditPolicy,
        value_head: ValueHead,
        env: SudokuMetaMDP,
        cfg: DictConfig,
        device: str = "cuda",
    ):
        """Initialize RL trainer.

        Args:
            model: TRM model wrapper
            policy: Edit policy network
            value_head: Value function network
            env: Meta-MDP environment
            cfg: Configuration
            device: Device for training
        """
        self.model = model
        self.policy = policy
        self.value_head = value_head
        self.env = env
        self.cfg = cfg
        self.device = device

        # RL config
        self.rl_cfg = cfg.rl if "rl" in cfg else OmegaConf.create()
        self.gamma = self.rl_cfg.get("gamma", 0.985)
        self.K = self.rl_cfg.get("K", 3)
        self.n_inner = self.rl_cfg.get("n_inner", 6)

        # Loss weights
        self.alpha_br = self.rl_cfg.get("alpha_br", 1.0)
        self.alpha_pi = self.rl_cfg.get("alpha_pi", 1.0)
        self.alpha_sup = self.rl_cfg.get("alpha_sup", 1.0)

        # Policy optimization
        self.ppo_clip = self.rl_cfg.get("ppo_clip", 0.2)
        self.entropy_beta = self.rl_cfg.get("entropy_beta", 0.001)

        # Spectral norm and Lipschitz monitoring
        self.use_spectral_norm = self.rl_cfg.get("spectral", {}).get("enabled", True)
        self.lipschitz_monitor = LipschitzMonitor(window_size=1000, ema_alpha=0.9)

        # Logging config
        self.log_Lz = self.rl_cfg.get("logging", {}).get("log_Lz", True)
        self.log_residuals = self.rl_cfg.get("logging", {}).get("log_residuals", True)

        # Apply spectral normalization if enabled
        if self.use_spectral_norm:
            self.model = apply_spectral_norm(
                self.model, name_patterns=["inner", "recurrent", "z_path"], n_power_iterations=1
            )

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.get("lr", 3e-4))
        self.value_optimizer = torch.optim.Adam(
            self.value_head.parameters(), lr=cfg.get("lr", 3e-4)
        )

        # Metrics
        self.step = 0
        self.epoch = 0

    def train_step(self, batch: dict) -> dict:
        """Perform one RL training step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        x = batch["input"].to(self.device)
        y = batch["output"].to(self.device)
        y_true = batch["target"].to(self.device)  # noqa: F841 - for future supervised loss

        batch_size = x.size(0)

        # Initialize z_0 (could be learned or zero)
        z = torch.zeros(batch_size, 128, device=self.device)  # Placeholder dim

        # ===== INNER UNROLL =====
        # Compute z_n = f_theta^{∘n}(z_0, y, x)
        with torch.no_grad():
            z_n = self.model.inner_unroll(z, y, x, n=self.n_inner)

        # ===== SAMPLE EDIT FROM POLICY =====
        # a ~ π_φ(·|y, z_n, x)
        action, old_log_prob = self.policy.sample(y, z_n, x)
        pos, val = action

        # Apply edit to get y'
        y_prime = y.clone()
        # y_prime[batch_idx, pos[batch_idx]] = val[batch_idx]  # Simplified
        # In practice, need proper edit application

        # ===== K-STEP ROLLOUT =====
        # Collect K-step trajectory with bootstrapped returns
        state_0 = (x, y_prime)

        # Compute returns G^(K)
        G_K, state_final = rollout_k(
            policy=self.policy,
            value_target=self.value_head.target_value,
            env=self.env,
            s0=state_0,
            K=self.K,
            gamma=self.gamma,
        )

        # ===== VALUE LOSS (BELLMAN RESIDUAL) =====
        # Compute current value estimate
        value_pred = self.value_head(z_n, x)

        # L_BR = (V(s) - stop_grad(G_K))^2
        value_loss = value_bellman_residual_loss(value_pred, G_K)

        # ===== ADVANTAGE COMPUTATION =====
        # A = G_K - V(s)
        advantages = compute_advantages(G_K, value_pred.detach())

        # Center advantages for stability
        advantages_centered = center_advantages(advantages)

        # ===== POLICY LOSS (PPO) =====
        # Recompute log prob under current policy
        new_log_prob = self.policy.log_prob(y, z_n, x, action)

        # Entropy (optional)
        # For factorized discrete policy, entropy can be computed from logits
        # entropy = ...  # Placeholder

        # PPO loss with clipping
        policy_loss = ppo_policy_loss(
            log_probs=new_log_prob,
            old_log_probs=old_log_prob.detach(),
            advantages=advantages_centered,
            epsilon=self.ppo_clip,
            entropy=None,  # TODO: compute entropy
            beta=self.entropy_beta,
        )

        # ===== SUPERVISED LOSS (OPTIONAL) =====
        # Mix in supervised CE loss if alpha_sup > 0
        supervised_loss = torch.tensor(0.0, device=self.device)
        if self.alpha_sup > 0:
            # pred = self.model(y, x)
            # supervised_loss = F.cross_entropy(pred, y_true)
            pass  # Placeholder

        # ===== TOTAL LOSS =====
        total_loss = (
            self.alpha_br * value_loss
            + self.alpha_pi * policy_loss
            + self.alpha_sup * supervised_loss
        )

        # ===== OPTIMIZATION STEP =====
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)

        self.value_optimizer.step()
        self.policy_optimizer.step()

        # ===== UPDATE TARGET NETWORK =====
        tau_ema = self.rl_cfg.get("tau_ema", 0.995)
        self.value_head.update_target(tau=tau_ema)

        # ===== COMPUTE METRICS =====
        metrics = {
            "loss/total": total_loss.item(),
            "loss/value": value_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/supervised": supervised_loss.item(),
            "policy/advantages_mean": advantages.mean().item(),
            "policy/advantages_std": advantages.std().item(),
            "value/returns_mean": G_K.mean().item(),
            "value/predictions_mean": value_pred.mean().item(),
        }

        # Lipschitz constant monitoring
        if self.log_Lz and self.step % 10 == 0:
            try:
                # Estimate Lz using Jacobian power iteration
                def f(z_in, y_in, x_in):
                    return self.model.inner_unroll(z_in, y_in, x_in, n=1)

                L_z, _ = estimate_Lz(f, z, y, x, iters=3)
                self.lipschitz_monitor.update(L_z)

                metrics["contraction/Lz"] = L_z
                metrics["contraction/Lz_ema"] = self.lipschitz_monitor.get_ema_lipschitz()
            except Exception as e:
                print(f"Warning: Failed to estimate Lz: {e}")

        # Bellman residuals
        if self.log_residuals:
            residuals = (value_pred - G_K.detach()).abs()
            metrics["value/residual_mean"] = residuals.mean().item()
            metrics["value/residual_max"] = residuals.max().item()

        # Task-specific metrics (Sudoku validity)
        # metrics["task/sudoku_validity"] = ...  # TODO

        self.step += 1
        return metrics

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        self.policy.train()
        self.value_head.train()

        epoch_metrics = {}
        num_batches = 0

        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            try:
                metrics = self.train_step(batch)

                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value

                num_batches += 1

                # Update progress bar
                if batch_idx % 10 == 0:
                    pbar.set_postfix(
                        {
                            "loss": f"{metrics['loss/total']:.4f}",
                            "V_loss": f"{metrics['loss/value']:.4f}",
                            "π_loss": f"{metrics['loss/policy']:.4f}",
                        }
                    )

                # Log to wandb
                if wandb.run is not None:
                    wandb.log({"step": self.step, **metrics})

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        self.epoch += 1
        return epoch_metrics

    def save_checkpoint(self, path: str):
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model": self.model.state_dict(),
            "policy": self.policy.state_dict(),
            "value_head": self.value_head.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def main(cfg: DictConfig):
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("RL Training for TRM with Meta-MDP")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))

    # Check if RL is enabled
    rl_enabled = cfg.get("rl", {}).get("enabled", False)
    if not rl_enabled:
        print("RL training is not enabled. Set +rl.enabled=true")
        return

    print("\n✓ RL training enabled")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize wandb (optional)
    if cfg.get("project_name"):
        wandb.init(
            project=cfg.project_name,
            name=cfg.get("run_name", "rl-training"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print("✓ Wandb initialized")

    # ===== DATA LOADING =====
    print("\nLoading data...")
    try:
        train_loader, train_metadata = create_dataloader(
            PretrainConfig(**OmegaConf.to_container(cfg, resolve=True)),
            split="train",
            rank=0,
            world_size=1,
        )
        print(f"✓ Data loaded: {train_metadata}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using dummy metadata for testing")
        train_metadata = PuzzleDatasetMetadata(vocab_size=11, seq_len=81, num_puzzle_identifiers=1)

        # Create dummy data loader for testing
        class DummyDataset:
            def __iter__(self):
                for _ in range(10):  # 10 dummy batches
                    yield {
                        "input": torch.randint(0, 11, (4, 81)),
                        "output": torch.randint(0, 11, (4, 81)),
                        "target": torch.randint(0, 11, (4, 81)),
                    }

        train_loader = DummyDataset()

    # ===== MODEL CREATION =====
    print("\nInitializing models...")

    # Base TRM model (placeholder - adapt to actual TRM)
    class DummyTRM(nn.Module):
        def __init__(self, hidden_dim=128):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, z, y, x):
            return torch.relu(self.fc(z))

    base_model = DummyTRM(hidden_dim=128).to(device)
    model = TRMWrapper(base_model, n_inner=cfg.rl.get("n_inner", 6))

    # Policy network
    policy = EditPolicy(
        z_dim=128,
        y_dim=81,  # Sudoku seq len
        num_cells=81,  # 9x9 grid
        num_vals=10,  # 1-9 + blank
        hidden=256,
    ).to(device)

    # Value network
    value_head = ValueHead(z_dim=128, x_dim=81, hidden=256).to(device)

    # Meta-MDP environment
    env = SudokuMetaMDP(edit_penalty=0.01)

    print(f"✓ Model initialized: {sum(p.numel() for p in model.parameters())} params")
    print(f"✓ Policy initialized: {sum(p.numel() for p in policy.parameters())} params")
    print(f"✓ Value head initialized: {sum(p.numel() for p in value_head.parameters())} params")

    # ===== TRAINER =====
    trainer = RLTrainer(
        model=model, policy=policy, value_head=value_head, env=env, cfg=cfg, device=device
    )

    print("\n" + "=" * 80)
    print("Starting RL training...")
    print("=" * 80)

    # ===== TRAINING LOOP =====
    num_epochs = cfg.get("epochs", 1)

    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")

        try:
            epoch_metrics = trainer.train_epoch(train_loader)

            print("\nEpoch metrics:")
            for key, value in epoch_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Log epoch metrics
            if wandb.run is not None:
                wandb.log({"epoch": epoch, **epoch_metrics})

            # Save checkpoint
            checkpoint_path = cfg.get("checkpoint_path", "checkpoints")
            os.makedirs(checkpoint_path, exist_ok=True)
            trainer.save_checkpoint(os.path.join(checkpoint_path, f"rl_epoch_{epoch}.pt"))

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            break
        except Exception as e:
            print(f"\n\nError in epoch {epoch}: {e}")
            import traceback

            traceback.print_exc()
            break

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
