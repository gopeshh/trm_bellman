#!/usr/bin/env python3
"""
Simple REINFORCE trainer for Sudoku - bypassing all the complex UPI machinery.
Just basic policy gradient to verify the environment and policy head work.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np
from pathlib import Path

# ===== Simple Policy Network =====
class SimpleSudokuPolicy(nn.Module):
    """
    Very simple policy network for 4x4 Sudoku.
    Input: flattened puzzle state (16 cells, vocab 6)
    Output: logits over 28 valid edit actions
    """
    def __init__(self, seq_len=16, vocab_size=6, num_actions=28, hidden_size=128):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        
        self.embed = nn.Embedding(vocab_size, 32)
        self.fc1 = nn.Linear(seq_len * 32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x, action_mask=None):
        """
        x: (batch, seq_len) long tensor of cell values
        action_mask: (batch, num_actions) bool tensor, True = valid action
        Returns: Categorical distribution over actions
        """
        # Embed and flatten
        h = self.embed(x)  # (B, seq_len, 32)
        h = h.view(h.size(0), -1)  # (B, seq_len * 32)
        
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        logits = self.fc3(h)  # (B, num_actions)
        
        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        return Categorical(logits=logits)


# ===== Environment Helper =====
class Simple4x4SudokuEnv:
    """
    Simplified 4x4 Sudoku environment with FIXED action space.
    State: 16-element array (vocab: 0=pad, 1=empty, 2-5=digits)
    Action: FIXED mapping (cell_position, digit) -> action_id
    
    CRITICAL: Action space is FIXED at 16 cells x 4 digits = 64 actions.
    Actions for given cells are masked out.
    """
    def __init__(self, puzzles, solutions):
        self.puzzles = puzzles  # List of (16,) arrays
        self.solutions = solutions  # List of (16,) arrays
        self.grid_size = 4
        self.seq_len = 16
        self.vocab_size = 6  # 0=pad, 1=empty, 2-5 = digits 1-4
        self.num_digits = 4
        self.num_actions = self.seq_len * self.num_digits  # 64 actions
        
    def reset(self, puzzle_idx=None):
        if puzzle_idx is None:
            puzzle_idx = random.randint(0, len(self.puzzles) - 1)
        self.current_puzzle = self.puzzles[puzzle_idx].copy()
        self.current_solution = self.solutions[puzzle_idx].copy()
        self.current_state = self.current_puzzle.copy()
        self.puzzle_idx = puzzle_idx
        
        # Store which cells are "given" (fixed at reset)
        self.given_mask = self.current_puzzle != 1  # True for given cells
        
        return self.current_state.copy()
    
    def create_action_mask(self, num_total_actions=64):
        """
        Create a boolean mask for valid actions.
        FIXED action space: action_id = cell_pos * 4 + (digit - 1)
        Invalid actions: editing given cells, or filling already-filled cells
        """
        mask = torch.zeros(num_total_actions, dtype=torch.bool)
        
        for pos in range(self.seq_len):
            # Can't edit given cells
            if self.given_mask[pos]:
                continue
            
            # Can only edit if cell is currently empty
            if self.current_state[pos] != 1:
                continue
            
            # This cell is valid to edit - all 4 digits are valid actions
            for d in range(self.num_digits):
                action_id = pos * self.num_digits + d
                if action_id < num_total_actions:
                    mask[action_id] = True
        
        return mask
    
    def decode_action(self, action_id):
        """
        Decode action_id to (cell_pos, digit).
        FIXED mapping: action_id = cell_pos * 4 + (digit - 1)
        """
        cell_pos = action_id // self.num_digits
        digit = (action_id % self.num_digits) + 1  # digit 1-4
        return cell_pos, digit
    
    def step(self, action_id):
        """
        Take action and return (next_state, reward, done).
        Reward: +1 if correct digit, 0 if wrong digit (sparse).
        Done: True if all cells filled.
        """
        cell_pos, digit = self.decode_action(action_id)
        
        # Validate action
        if cell_pos >= self.seq_len:
            return self.current_state.copy(), 0.0, True
        
        if self.given_mask[cell_pos]:
            # Can't edit given cells - no-op
            return self.current_state.copy(), 0.0, False
        
        # Apply action
        digit_vocab = digit + 1  # map digit 1-4 to vocab 2-5
        self.current_state[cell_pos] = digit_vocab
        
        # Check if correct
        correct_vocab = self.current_solution[cell_pos]
        if digit_vocab == correct_vocab:
            reward = 1.0
        else:
            reward = 0.0  # Sparse: only reward correct
        
        # Check if done (all non-given cells filled)
        empty_count = (self.current_state == 1).sum()
        done = empty_count == 0
        
        return self.current_state.copy(), reward, done
    
    def compute_score(self):
        """Compute current score (fraction of cells matching solution)"""
        non_given_mask = ~self.given_mask
        
        # Only score non-given cells
        correct = (self.current_state == self.current_solution) & non_given_mask
        total_non_given = non_given_mask.sum()
        
        if total_non_given == 0:
            return 1.0
        return correct.sum() / total_non_given


def load_puzzles(dataset_path):
    """Load puzzles from the dataset directory (numpy format)."""
    
    dataset_path = Path(dataset_path)
    
    # Load from train/ subdirectory
    train_path = dataset_path / "train"
    
    inputs_path = train_path / "all__inputs.npy"
    labels_path = train_path / "all__labels.npy"
    
    if not inputs_path.exists():
        raise FileNotFoundError(f"Could not find {inputs_path}")
    
    inputs = np.load(inputs_path)  # shape: (N, seq_len)
    labels = np.load(labels_path)  # shape: (N, seq_len)
    
    # Convert to list of puzzles
    puzzles = [inputs[i] for i in range(len(inputs))]
    solutions = [labels[i] for i in range(len(labels))]
    
    return puzzles, solutions


class ValueNetwork(nn.Module):
    """Simple baseline network for variance reduction."""
    def __init__(self, seq_len=16, vocab_size=6, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.fc1 = nn.Linear(seq_len * 32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h = self.embed(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


def train_simple_reinforce(
    dataset_path: str,
    num_episodes: int = 10000,
    lr: float = 0.001,
    gamma: float = 0.99,
    eval_interval: int = 500,
    seed: int = 42,
):
    """Train with Actor-Critic REINFORCE with baseline."""
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    puzzles, solutions = load_puzzles(dataset_path)
    print(f"Loaded {len(puzzles)} puzzles from {dataset_path}")
    
    if len(puzzles) == 0:
        print("No puzzles found!")
        return
    
    # Create environment and policy
    env = Simple4x4SudokuEnv(puzzles, solutions)
    
    # Calculate action space - FIXED at 16 cells x 4 digits = 64
    sample_state = env.reset(0)
    num_empty = (sample_state == 1).sum()
    print(f"First puzzle has {num_empty} empty cells")
    
    # Fixed action space size: 16 cells * 4 digits = 64
    MAX_ACTIONS = 64
    print(f"Fixed action space: {MAX_ACTIONS} actions")
    
    policy = SimpleSudokuPolicy(seq_len=16, vocab_size=6, num_actions=MAX_ACTIONS, hidden_size=256)
    value_net = ValueNetwork(seq_len=16, vocab_size=6, hidden_size=256)
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr * 2)
    
    # Training stats
    episode_rewards = []
    episode_lengths = []
    solve_count = 0
    
    for ep in range(num_episodes):
        # Reset environment
        state = env.reset()
        
        log_probs = []
        rewards = []
        states = []
        values = []
        
        # Rollout
        done = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.long).unsqueeze(0)
            action_mask = env.create_action_mask(MAX_ACTIONS).unsqueeze(0)
            
            # Get value estimate
            with torch.no_grad():
                value = value_net(state_t)
            values.append(value.item())
            states.append(state.copy())
            
            # Get action distribution
            dist = policy(state_t, action_mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Take step
            next_state, reward, done = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        values_t = torch.tensor(values)
        
        # Compute advantages (returns - baseline)
        advantages = returns - values_t
        
        # Update value network
        states_t = torch.tensor(np.array(states), dtype=torch.long)
        values_pred = value_net(states_t)
        value_loss = F.mse_loss(values_pred, returns)
        
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy gradient loss with advantage
        policy_loss = 0
        for log_p, adv in zip(log_probs, advantages):
            policy_loss -= log_p * adv.detach()
        
        # Add entropy bonus for exploration
        entropy = 0
        for state_s in states:
            state_t = torch.tensor(state_s, dtype=torch.long).unsqueeze(0)
            env.current_state = state_s.copy()  # restore state for mask
            action_mask = env.create_action_mask(MAX_ACTIONS).unsqueeze(0)
            dist = policy(state_t, action_mask)
            entropy += dist.entropy().mean()
        entropy_coef = 0.01
        policy_loss -= entropy_coef * entropy
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_optimizer.step()
        
        # Track stats
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        episode_lengths.append(len(rewards))
        
        # Check if solved
        if env.compute_score() == 1.0:
            solve_count += 1
        
        # Evaluation
        if (ep + 1) % eval_interval == 0:
            # Greedy evaluation
            eval_solves = 0
            eval_scores = []
            
            for puzzle_idx in range(len(puzzles)):
                state = env.reset(puzzle_idx)
                for _ in range(20):  # max steps
                    state_t = torch.tensor(state, dtype=torch.long).unsqueeze(0)
                    action_mask = env.create_action_mask(MAX_ACTIONS).unsqueeze(0)
                    
                    with torch.no_grad():
                        dist = policy(state_t, action_mask)
                        action = dist.probs.argmax()  # greedy
                    
                    next_state, _, done = env.step(action.item())
                    state = next_state
                    if done:
                        break
                
                score = env.compute_score()
                eval_scores.append(score)
                if score == 1.0:
                    eval_solves += 1
            
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_len = np.mean(episode_lengths[-eval_interval:])
            train_solve_rate = solve_count / eval_interval
            solve_count = 0
            
            print(f"[Episode {ep+1}] "
                  f"train_reward={avg_reward:.2f} train_len={avg_len:.1f} "
                  f"train_solve={train_solve_rate:.2%} | "
                  f"eval_solve={eval_solves}/{len(puzzles)} "
                  f"eval_score_mean={np.mean(eval_scores):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-paths", type=str, default="data/sudoku-4x4-ultra-easy")
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_simple_reinforce(
        dataset_path=args.dataset_paths,
        num_episodes=args.num_episodes,
        lr=args.lr,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )

