"""
run_demo.py
===========
Minimal entry point for the options market-making simulator.

Always runs: one representative simulation (SimConfig(seed=0), using
every other SimConfig default), prints its summarize_run() metrics,
and saves its diagnostic plot to output/single_run.png.

Optionally (--experiments flag): also runs all five parameter sweeps
via run_all_experiments() and saves each sweep's RAW result DataFrame
as a CSV in output/ -- no sweep plotting, no aggregation, no new
analysis. Every number here comes from run_simulation(),
summarize_run(), plot_simulation(), or run_all_experiments() -- this
script contains no simulation, statistical, or trading logic of its
own.

Usage:
    python run_demo.py                 # single run + plot only
    python run_demo.py --experiments   # also runs all five sweeps
"""

import argparse
import os

from src.simulation import SimConfig, run_simulation
from src.analytics import summarize_run, plot_simulation
from src.experiments import run_all_experiments

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main(run_experiments: bool = False, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    print("Running a single simulation (SimConfig(seed=0))...")
    config = SimConfig(seed=0)
    history, pnl_history = run_simulation(config)

    summary = summarize_run(history, pnl_history)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    plot_path = os.path.join(out_dir, "single_run.png")
    plot_simulation(history, pnl_history,
                     title="Options Market Maker -- Single Run",
                     save_path=plot_path)
    print(f"\nSaved diagnostic plot to {plot_path}")

    if run_experiments:
        print("\nRunning all five parameter sweeps...")
        results = run_all_experiments(n_repeats=5)
        for param_name, df in results.items():
            csv_path = os.path.join(out_dir, f"sweep_{param_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  {param_name}: {len(df)} rows -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options market-making simulator demo")
    parser.add_argument("--experiments", action="store_true",
                         help="also run all five parameter sweeps and save their CSVs")
    args = parser.parse_args()
    main(run_experiments=args.experiments)