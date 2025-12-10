#!/usr/bin/env python3
"""
Imitation learning from oracle for Sudoku.
This verifies the model architecture can learn Sudoku before we try RL.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path

# Simple policy network (same as before but with better architecture)
class SudokuPolicyNet(nn.Module):
    def __init__(self, seq_len=16, vocab_size=6, num_actions=97, hidden_size=256):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        
        # Embed each cell
        self.embed = nn.Embedding(vocab_size, 64)
        
        # Process with transformer-like attention
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, 64) * 0.02)
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        
        self.ffn = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
        )
        
        # Output policy over actions
        self.fc_out = nn.Linear(seq_len * 64, num_actions)
    
    def forward(self, x, action_mask=None):
        """
        x: (batch, seq_len) long tensor of cell values
        action_mask: (batch, num_actions) bool tensor, True = valid action
        """
        # Embed
        h = self.embed(x) + self.pos_embed  # (B, seq_len, 64)
        
        # Self-attention
        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        
        # Flatten and project to actions
        h = h.reshape(h.size(0), -1)  # (B, seq_len * 64)
        logits = self.fc_out(h)  # (B, num_actions)
        
        # Mask invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        return logits


def load_puzzles(dataset_path):
    """Load puzzles from the dataset directory (numpy format)."""
    dataset_path = Path(dataset_path)
    train_path = dataset_path / "train"
    
    inputs = np.load(train_path / "all__inputs.npy")
    labels = np.load(train_path / "all__labels.npy")
    
    return inputs, labels


def compute_action_mask(inputs, given_mask, vocab_size=6, stop_action_id=96):
    """
    Compute action mask for valid edits.
    Actions: pos * vocab_size + tok for pos in 0..15, tok in 0..5
    Plus STOP action at stop_action_id.
    
    Valid actions: editing non-given cells with tokens 2-5 (digits 1-4).
    """
    seq_len = 16
    num_actions = stop_action_id + 1  # 97
    
    mask = torch.zeros(num_actions, dtype=torch.bool)
    
    for pos in range(seq_len):
        if given_mask[pos]:
            continue  # can't edit given cells
        
        # Allow tokens 2-5 (digits 1-4)
        for tok in range(2, vocab_size):
            action_id = pos * vocab_size + tok
            mask[action_id] = True
    
    # STOP action disabled
    mask[stop_action_id] = False
    
    return mask


def get_oracle_action(pos, solution, vocab_size=6):
    """Get the oracle action for filling position pos correctly."""
    correct_vocab = solution[pos]
    action_id = pos * vocab_size + correct_vocab
    return action_id


def train_imitation(dataset_path, num_epochs=100, lr=0.001, seed=42):
    """Train policy via imitation learning from oracle."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    inputs, labels = load_puzzles(dataset_path)
    num_puzzles = len(inputs)
    print(f"Loaded {num_puzzles} puzzles")
    
    # Setup
    vocab_size = 6
    seq_len = 16
    stop_action_id = seq_len * vocab_size  # 96
    num_actions = stop_action_id + 1  # 97
    
    policy = SudokuPolicyNet(seq_len, vocab_size, num_actions, hidden_size=256)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    print(f"\nTraining imitation learning for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_steps = 0
        
        # Shuffle puzzles
        indices = np.random.permutation(num_puzzles)
        
        for puzzle_idx in indices:
            puzzle = inputs[puzzle_idx].copy()
            solution = labels[puzzle_idx]
            given_mask = puzzle != 1  # given cells have value > 1
            
            # Simulate episode - fill each empty cell in order
            empty_positions = [i for i in range(seq_len) if puzzle[i] == 1]
            
            for pos in empty_positions:
                # Current state
                state = torch.tensor(puzzle, dtype=torch.long).unsqueeze(0)
                
                # Create mask (positions not yet filled)
                current_given = puzzle != 1
                mask = compute_action_mask(puzzle, current_given, vocab_size, stop_action_id)
                mask = mask.unsqueeze(0)
                
                # Get oracle action
                target_action = get_oracle_action(pos, solution, vocab_size)
                
                # Forward pass
                logits = policy(state, mask)
                target = torch.tensor([target_action], dtype=torch.long)
                
                # Cross-entropy loss
                loss = F.cross_entropy(logits, target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Check if prediction is correct
                pred = logits.argmax(1).item()
                if pred == target_action:
                    total_correct += 1
                total_steps += 1
                
                # Apply oracle action to state for next step
                correct_vocab = solution[pos]
                puzzle[pos] = correct_vocab
        
        # Report
        accuracy = total_correct / total_steps if total_steps > 0 else 0
        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.2%}")
    
    # Final evaluation
    print("\nFinal evaluation (greedy rollouts):")
    eval_solve = 0
    eval_scores = []
    
    for puzzle_idx in range(min(50, num_puzzles)):
        puzzle = inputs[puzzle_idx].copy()
        solution = labels[puzzle_idx]
        given_mask = puzzle != 1
        
        # Greedy rollout
        for _ in range(10):  # max steps
            empty_positions = [i for i in range(seq_len) if puzzle[i] == 1]
            if len(empty_positions) == 0:
                break
            
            state = torch.tensor(puzzle, dtype=torch.long).unsqueeze(0)
            current_given = puzzle != 1
            mask = compute_action_mask(puzzle, current_given, vocab_size, stop_action_id)
            mask = mask.unsqueeze(0)
            
            with torch.no_grad():
                logits = policy(state, mask)
                action = logits.argmax(1).item()
            
            # Decode and apply action
            pos = action // vocab_size
            tok = action % vocab_size
            
            if pos < seq_len and puzzle[pos] == 1:
                puzzle[pos] = tok
        
        # Compute score
        non_given = ~given_mask
        correct = (puzzle == solution) & non_given
        total_empty = non_given.sum()
        score = correct.sum() / total_empty if total_empty > 0 else 1.0
        eval_scores.append(score)
        
        if score == 1.0:
            eval_solve += 1
    
    print(f"Eval: solved={eval_solve}/{min(50, num_puzzles)}, mean_score={np.mean(eval_scores):.3f}")
    
    return policy


def demo_solve(policy, inputs, labels, puzzle_idx=0):
    """Demo: show step-by-step solving of a puzzle."""
    vocab_size = 6
    seq_len = 16
    stop_action_id = seq_len * vocab_size
    
    puzzle = inputs[puzzle_idx].copy()
    solution = labels[puzzle_idx]
    given_mask = puzzle != 1
    
    print("\n" + "="*60)
    print(f"DEMO: Solving puzzle {puzzle_idx}")
    print("="*60)
    
    def format_grid(arr):
        """Format 4x4 grid for display."""
        lines = []
        for row in range(4):
            cells = []
            for col in range(4):
                val = arr[row * 4 + col]
                if val == 1:
                    cells.append(".")
                else:
                    cells.append(str(val - 1))  # vocab 2-5 -> digit 1-4
            lines.append(" ".join(cells))
        return "\n".join(lines)
    
    print("\nInitial puzzle:")
    print(format_grid(puzzle))
    print("\nTarget solution:")
    print(format_grid(solution))
    
    print("\n--- Solving step by step ---")
    
    for step in range(10):
        empty_positions = [i for i in range(seq_len) if puzzle[i] == 1]
        if len(empty_positions) == 0:
            break
        
        state = torch.tensor(puzzle, dtype=torch.long).unsqueeze(0)
        current_given = puzzle != 1
        mask = compute_action_mask(puzzle, current_given, vocab_size, stop_action_id)
        mask = mask.unsqueeze(0)
        
        with torch.no_grad():
            logits = policy(state, mask)
            probs = F.softmax(logits, dim=-1)
            action = logits.argmax(1).item()
        
        pos = action // vocab_size
        tok = action % vocab_size
        row, col = pos // 4, pos % 4
        digit = tok - 1  # vocab 2-5 -> digit 1-4
        
        correct_tok = solution[pos]
        is_correct = tok == correct_tok
        
        print(f"Step {step+1}: Place digit {digit} at ({row},{col}) {'✓' if is_correct else '✗'}")
        
        if pos < seq_len and puzzle[pos] == 1:
            puzzle[pos] = tok
    
    print("\n--- Final result ---")
    print(format_grid(puzzle))
    
    # Check if solved
    non_given = ~given_mask
    correct = (puzzle == solution) & non_given
    solved = correct.sum() == non_given.sum()
    
    print(f"\n{'SOLVED!' if solved else 'NOT SOLVED'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-paths", type=str, default="data/sudoku-4x4-ultra-easy")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Run demo after training")
    parser.add_argument("--save", type=str, help="Save model to path")
    args = parser.parse_args()
    
    policy = train_imitation(
        dataset_path=args.dataset_paths,
        num_epochs=args.num_epochs,
        lr=args.lr,
        seed=args.seed,
    )
    
    # Demo (always run to show learned behavior)
    if True:
        inputs, labels = load_puzzles(args.dataset_paths)
        for i in range(3):
            demo_solve(policy, inputs, labels, puzzle_idx=i)
    
    # Save
    if args.save:
        torch.save(policy.state_dict(), args.save)
        print(f"\nModel saved to {args.save}")

