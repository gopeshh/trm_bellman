#!/usr/bin/env python3
"""Backward-compatible wrapper for the simple RL training entrypoint."""

from entrypoints.simple_rl_train import main


if __name__ == "__main__":
    raise SystemExit(main())
