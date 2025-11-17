import argparse
from typing import List, Optional, Tuple

import torch

try:
    from tqdm import trange
except ImportError:  # pragma: no cover
    trange = None

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # type: ignore
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from rl.config import RLConfig
from rl.envs.plan_edit_env import PlanEditEnv, PlanEditEnvConfig
from rl.upi_trm_trainer import UPITrmTrainer
from utils.seeding import set_global_seed


def dummy_checker(x, y) -> float:
    """
    Placeholder checker: reward is negative L1 distance between plan and inputs.
    """
    target = x["inputs"]
    plan = y if torch.is_tensor(y) else torch.as_tensor(y)
    target = target.to(torch.float32)
    plan = plan.to(torch.float32)
    return float(-(target - plan).abs().mean().item())


class DummyPuzzleDataset:
    """
    Tiny in-memory dataset suitable for smoke tests of the RL loop.
    """

    def __init__(self, num_instances: int = 32, seq_len: int = 16, vocab_size: int = 32):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_identifiers = num_instances

        self.samples = []
        for idx in range(num_instances):
            inputs = torch.randint(low=0, high=vocab_size, size=(seq_len,), dtype=torch.long)
            puzzle_identifier = torch.tensor(idx, dtype=torch.long)
            self.samples.append(
                {
                    "inputs": inputs,
                    "puzzle_identifiers": puzzle_identifier,
                    "initial_plan": torch.zeros_like(inputs),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class OfflinePuzzleDataset:
    """
    Wraps a finite list of samples gathered from PuzzleDataset to provide __len__/__getitem__.
    """

    def __init__(self, samples: List[dict], seq_len: int, vocab_size: int, num_identifiers: int):
        self.samples = samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_identifiers = num_identifiers

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def build_dataset_from_paths(
    dataset_paths: Optional[List[str]],
    pool_size: int,
) -> Tuple[object, int, int, int]:
    """
    Attempt to load a handful of samples from the supervised PuzzleDataset to bootstrap RL.
    Falls back to DummyPuzzleDataset if paths are missing or loading fails.
    """

    if dataset_paths:
        try:
            ds_cfg = PuzzleDatasetConfig(
                seed=0,
                dataset_paths=dataset_paths,
                global_batch_size=pool_size,
                test_set_mode=True,
                epochs_per_iter=1,
                rank=0,
                num_replicas=1,
            )
            iterable = PuzzleDataset(ds_cfg, split="train")
            samples: List[dict] = []
            for _set_name, batch, _ in iterable:
                batch_inputs = batch["inputs"]
                batch_ids = batch["puzzle_identifiers"]
                batch_size = batch_inputs.shape[0]
                for i in range(batch_size):
                    inputs = batch_inputs[i].clone()
                    puzzle_id = batch_ids[i].clone()
                    samples.append(
                        {
                            "inputs": inputs,
                            "puzzle_identifiers": puzzle_id,
                            "initial_plan": torch.zeros_like(inputs),
                        }
                    )
                    if len(samples) >= pool_size:
                        break
                if len(samples) >= pool_size:
                    break
            if samples:
                num_identifiers = int(torch.stack([s["puzzle_identifiers"] for s in samples]).max().item() + 1)
                dataset = OfflinePuzzleDataset(
                    samples=samples,
                    seq_len=samples[0]["inputs"].shape[-1],
                    vocab_size=iterable.metadata.vocab_size,
                    num_identifiers=max(num_identifiers, iterable.metadata.num_puzzle_identifiers),
                )
                return dataset, dataset.seq_len, dataset.vocab_size, dataset.num_identifiers
        except Exception as exc:  # pragma: no cover - best-effort bootstrap
            print(f"[upi_trm_train] Falling back to dummy dataset (reason: {exc})")

    dummy = DummyPuzzleDataset()
    return dummy, dummy.seq_len, dummy.vocab_size, dummy.num_identifiers


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyRecursiveReasoningModel with plan-space RL.")
    parser.add_argument("--dataset-paths", nargs="+", default=None, help="Optional list of supervised dataset directories.")
    parser.add_argument("--train-steps", type=int, default=200, help="Number of outer RL steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for TD updates.")
    parser.add_argument("--rollouts-per-step", type=int, default=1, help="Episodes collected before each optimization step.")
    parser.add_argument("--max-edits", type=int, default=8, help="Maximum edits per episode.")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval in train steps.")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval in train steps.")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Number of episodes per evaluation call.")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config overriding RLConfig defaults.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        set_global_seed(args.seed)

    rl_cfg = RLConfig(
        batch_size=args.batch_size,
        num_train_steps=args.train_steps,
        rollout_episodes_per_step=args.rollouts_per_step,
        max_edits=args.max_edits,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_num_episodes=args.eval_episodes,
        use_tqdm=not args.no_tqdm,
    )

    if args.config is not None:
        import yaml

        with open(args.config, "r") as f:
            override = yaml.safe_load(f) or {}
        rl_cfg = RLConfig(**{**rl_cfg.dict(), **override})

    dataset, seq_len, vocab_size, num_identifiers = build_dataset_from_paths(
        dataset_paths=args.dataset_paths,
        pool_size=max(rl_cfg.batch_size, 8),
    )

    env_cfg = PlanEditEnvConfig(
        max_edits=rl_cfg.max_edits,
        gamma=rl_cfg.gamma,
        reward_shaping=True,
    )
    env = PlanEditEnv(dataset=dataset, checker=dummy_checker, config=env_cfg)

    rl_num_actions = 4  # placeholder discrete action space (3 edits + STOP)
    env.set_stop_action_id(stop_id=rl_num_actions - 1)

    trm_cfg_dict = dict(
        batch_size=rl_cfg.batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=max(num_identifiers, rl_cfg.batch_size),
        vocab_size=vocab_size,
        H_cycles=2,
        L_cycles=2,
        H_layers=0,
        L_layers=1,
        hidden_size=64,
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
        rl_enable_contraction=rl_cfg.enable_contraction,
        rl_enable_policy_head=True,
        rl_num_actions=rl_num_actions,
    )

    model = TinyRecursiveReasoningModel_ACTV1(trm_cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = UPITrmTrainer(model=model, env=env, rl_cfg=rl_cfg, device=device)

    if rl_cfg.use_tqdm and trange is not None:
        step_iter = trange(rl_cfg.num_train_steps, desc="UPI-TRM RL training")
    else:
        step_iter = range(rl_cfg.num_train_steps)

    for step in step_iter:
        metrics = trainer.train_step()
        if (step + 1) % rl_cfg.log_interval == 0:
            msg = (
                f"[step {step+1:05d}] value_loss={metrics['loss_value']:.6f} "
                f"policy_loss={metrics['loss_policy']:.6f}"
            )
            if hasattr(step_iter, "write"):
                step_iter.write(msg)
            else:
                print(msg)

        if (step + 1) % rl_cfg.eval_interval == 0:
            success_rate = trainer.evaluate_policy_success_rate(env_cfg=env_cfg, dataset=dataset, checker=dummy_checker)
            eval_msg = f"[step {step+1:05d}] eval_success_rate={success_rate:.3f}"
            if hasattr(step_iter, "write"):
                step_iter.write(eval_msg)
            else:
                print(eval_msg)


if __name__ == "__main__":
    main()

