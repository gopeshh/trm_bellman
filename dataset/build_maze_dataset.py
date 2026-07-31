import argparse
from typing import Optional
import math
import os
import csv
import json
import numpy as np

from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from dataset.common import PuzzleDatasetMetadata, dihedral_transform


CHARSET = "# SGo"


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/maze-30x30-hard-1k"
    source_revision: Optional[str] = None
    output_dir: str = "data/maze-30x30-hard-1k"
    seed: int = 42

    subsample_size: Optional[int] = None
    aug: bool = False


def convert_subset(set_name: str, config: DataProcessConfig):
    rng = np.random.default_rng(config.seed + (0 if set_name == "train" else 1))
    # Read CSV
    all_chars = set()
    grid_size = None
    inputs = []
    labels = []
    
    with open(
        hf_hub_download(
            config.source_repo,
            f"{set_name}.csv",
            repo_type="dataset",
            revision=config.source_revision,
        ),
        newline="",
    ) as csvfile:  # type: ignore
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            all_chars.update(q)
            all_chars.update(a)

            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)
                
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = rng.choice(
                total_samples,
                size=config.subsample_size,
                replace=False,
            )
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    result_inputs = []
    result_labels = []
    puzzle_identifiers = []
    puzzle_indices = [0]
    group_indices = [0]
    puzzle_id = 0
    example_id = 0
    
    for inp, out in zip(tqdm(inputs), labels):
        # Dihedral transformations for augmentation
        for aug_idx in range(8 if (set_name == "train" and config.aug) else 1):
            result_inputs.append(dihedral_transform(inp, aug_idx))
            result_labels.append(dihedral_transform(out, aug_idx))
            example_id += 1
            puzzle_id += 1
            
            puzzle_indices.append(example_id)
            puzzle_identifiers.append(0)
            
        # Push group
        group_indices.append(puzzle_id)
            
    # Char mappings
    assert len(all_chars - set(CHARSET)) == 0
    
    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.vstack([char2id[s.reshape(-1)] for s in seq])
        
        return arr
    
    results = {
        "inputs": _seq_to_numpy(result_inputs),
        "labels": _seq_to_numpy(result_labels),
        
        "group_indices": np.array(group_indices, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),  # type: ignore
        vocab_size=len(CHARSET) + 1,  # PAD + Charset
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def preprocess_data(config: DataProcessConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "build_config.json"), "w") as f:
        payload = config.model_dump(exclude={"output_dir"})
        json.dump(payload, f, sort_keys=True)
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a maze puzzle dataset.")
    parser.add_argument(
        "--source-repo", default="sapientinc/maze-30x30-hard-1k"
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--output-dir", default="data/maze-30x30-hard-1k")
    parser.add_argument("--subsample-size", type=int)
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    preprocess_data(DataProcessConfig(**vars(parser.parse_args())))
