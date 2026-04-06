#!/bin/bash
set -euo pipefail

FBROOT="${FBROOT:-$HOME/fbsource/fbcode}"

if [ ! -d "$FBROOT/buiksat_trm" ]; then
  echo "Expected repo at $FBROOT/buiksat_trm" >&2
  exit 1
fi

cd "$FBROOT"

echo "============================================================"
echo "Restoring 4x4 Sudoku datasets"
echo "============================================================"
echo "fbcode root: $FBROOT"
echo ""
echo "Datasets:"
echo "  - sudoku-4x4-trivial          (true ultra-easy, 1-4 empties, 500 puzzles)"
echo "  - sudoku-4x4-ultra-easy       (legacy compatibility path, 6-8 empties, 500 puzzles)"
echo "  - sudoku-4x4-easy_6to8empties (paper hard split, 6-8 empties, 1000 puzzles)"
echo ""

buck2 run //buiksat_trm:build_4x4_trivial -- \
  --output-dir buiksat_trm/data/sudoku-4x4-trivial \
  --num-puzzles 500 \
  --seed 42

buck2 run //buiksat_trm:build_4x4_sudoku -- \
  --output-dir buiksat_trm/data/sudoku-4x4-ultra-easy \
  --num-easy 500 \
  --num-medium 0 \
  --num-hard 0 \
  --seed 42

buck2 run //buiksat_trm:build_4x4_sudoku -- \
  --output-dir buiksat_trm/data/sudoku-4x4-easy_6to8empties \
  --num-easy 1000 \
  --num-medium 0 \
  --num-hard 0 \
  --seed 42

echo ""
echo "Validating regenerated datasets..."
buck2 run //buiksat_trm:inspect_4x4_dataset -- --datasets \
  buiksat_trm/data/sudoku-4x4-trivial \
  buiksat_trm/data/sudoku-4x4-ultra-easy \
  buiksat_trm/data/sudoku-4x4-easy_6to8empties
