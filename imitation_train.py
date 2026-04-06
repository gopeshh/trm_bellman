#!/usr/bin/env python3
"""Backward-compatible wrapper for the imitation training entrypoint."""

from entrypoints.imitation_train import main


if __name__ == "__main__":
    raise SystemExit(main())
