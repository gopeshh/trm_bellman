#!/usr/bin/env python3
"""
Backward-compatible outer-step view for the Table 3 baseline curves.

This keeps the pre-audit x-axis convention while reusing the same plotting
logic and method set, including the new DQN (n=5) baseline.
"""

from plot_table3_baselines import main


if __name__ == "__main__":
    main(
        default_x_axis_mode="outer",
        default_output_name="trivial_baselines_vs_no_contraction_success_vs_outer_steps",
        default_max_steps=80000,
    )
