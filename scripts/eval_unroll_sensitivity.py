#!/usr/bin/env python3
"""Backward-compatible wrapper for unroll sensitivity tooling."""

from scripts.eval.unroll_sensitivity import *  # noqa: F401,F403
from scripts.eval.unroll_sensitivity import main


if __name__ == "__main__":
    raise SystemExit(main())
