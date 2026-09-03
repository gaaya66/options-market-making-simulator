"""
tests/test_experiments.py
==========================
Unit tests for src/experiments.py: _run_sweep, the five sweep
functions, and run_all_experiments.

Run with:
    pytest tests/test_experiments.py
or directly:
    python tests/test_experiments.py

Scope: experiments.py is a thin orchestration layer over
run_simulation()/summarize_run() -- these tests focus on the sweep
MECHANICS (row counts, columns, reproducibility via common random
numbers, non-mutation of base_config, correct field propagation)
rather than re-verifying simulation/analytics correctness, which are
covered in their own test files. Configs here deliberately use small
n_steps and few values/repeats to keep the suite fast -- never to
change what's being verified.

Sections:
    1. _run_sweep shape and columns (via a sweep function)
    2. Reproducibility (common random numbers)
    3. base_config is never mutated
    4. Non-swept fields propagate through the sweep
    5. Each of the five sweep functions individually
    6. run_all_experiments
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy  # noqa: E402

import pandas as pd  # noqa: E402

from src.simulation import SimConfig  # noqa: E402
from src.experiments import (  # noqa: E402
    sweep_hedge_frequency, sweep_spread, sweep_volatility_mismatch,
    sweep_order_flow, sweep_inventory_limits, run_all_experiments,
)


def _small_base(**overrides):
    """A small, fast base config for testing sweep mechanics -- not
    meant to be a realistic research configuration."""
    defaults = dict(n_steps=20)
    defaults.update(overrides)
    return SimConfig(**defaults)


# ============================================================
# 1. _run_sweep shape and columns
# ============================================================

def test_sweep_row_count_is_values_times_repeats():
    df = sweep_hedge_frequency(_small_base(), values=(1, 5, 10), n_repeats=2)
    assert len(df) == 3 * 2


def test_sweep_includes_all_summarize_run_keys():
    df = sweep_hedge_frequency(_small_base(), values=(1,), n_repeats=1)
    expected = {"final_pnl", "final_option_pnl", "final_hedge_pnl", "sharpe",
                "max_drawdown", "hedging_error", "max_abs_inventory",
                "mean_abs_inventory", "n_hedge_trades",
                "mean_abs_gamma_exposure", "mean_abs_vega_exposure"}
    assert expected.issubset(set(df.columns))


def test_sweep_includes_param_and_rep_columns():
    df = sweep_hedge_frequency(_small_base(), values=(1, 5), n_repeats=2)
    assert "hedge_every" in df.columns
    assert "rep" in df.columns


def test_sweep_param_column_matches_input_values_repeated():
    df = sweep_hedge_frequency(_small_base(), values=(1, 5, 10), n_repeats=3)
    assert sorted(df["hedge_every"].unique().tolist()) == [1, 5, 10]
    assert (df["hedge_every"] == 1).sum() == 3
    assert (df["hedge_every"] == 5).sum() == 3
    assert (df["hedge_every"] == 10).sum() == 3


def test_sweep_rep_column_cycles_through_range():
    df = sweep_hedge_frequency(_small_base(), values=(1, 5), n_repeats=3)
    reps_for_value_1 = sorted(df[df["hedge_every"] == 1]["rep"].tolist())
    assert reps_for_value_1 == [0, 1, 2]


# ============================================================
# 2. Reproducibility (common random numbers)
# ============================================================

def test_sweep_is_reproducible():
    base = _small_base()
    df_a = sweep_hedge_frequency(base, values=(1, 5), n_repeats=2)
    df_b = sweep_hedge_frequency(base, values=(1, 5), n_repeats=2)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_repeat_zero_agrees_across_different_n_repeats():
    """Seed depends only on `rep`, not on how many repeats follow it --
    repeat 0's results with n_repeats=1 should exactly match repeat
    0's results with n_repeats=3."""
    base = _small_base()
    df_few = sweep_hedge_frequency(base, values=(1, 5), n_repeats=1)
    df_many = sweep_hedge_frequency(base, values=(1, 5), n_repeats=3)
    rep0_few = df_few[df_few["rep"] == 0].reset_index(drop=True)
    rep0_many = df_many[df_many["rep"] == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(rep0_few, rep0_many)


def test_common_random_numbers_same_seed_across_swept_values():
    """The defining property of common random numbers: two different
    swept parameter values, same repeat, must use the exact same
    underlying price path and order flow -- verified here indirectly
    via the S-path-dependent 'final_pnl' NOT being required to match
    (different hedge_every legitimately changes P&L), but confirmed
    directly at the simulation level (see test_simulation.py) and
    re-confirmed here structurally: seed only depends on rep."""
    from src.simulation import run_simulation
    base = _small_base()
    cfg_a = copy.deepcopy(base)
    cfg_a.hedge_every = 1
    cfg_a.seed = 0
    cfg_b = copy.deepcopy(base)
    cfg_b.hedge_every = 10
    cfg_b.seed = 0
    hist_a, _ = run_simulation(cfg_a)
    hist_b, _ = run_simulation(cfg_b)
    assert hist_a["S"].tolist() == hist_b["S"].tolist()


# ============================================================
# 3. base_config is never mutated
# ============================================================

def test_sweep_does_not_mutate_base_config():
    base = _small_base(hedge_every=1, mm_half_spread=0.05)
    snapshot = copy.deepcopy(base)
    sweep_hedge_frequency(base, values=(1, 5, 10), n_repeats=2)
    assert base == snapshot


def test_sweep_does_not_mutate_base_config_across_multiple_sweeps():
    base = _small_base()
    snapshot = copy.deepcopy(base)
    sweep_spread(base, values=(0.02, 0.05), n_repeats=1)
    sweep_order_flow(base, values=(0.5, 1.0), n_repeats=1)
    assert base == snapshot


# ============================================================
# 4. Non-swept fields propagate through the sweep
# ============================================================

def test_non_swept_fields_actually_propagate():
    """Two base configs differing only in mm_half_spread, swept over
    hedge_every, should produce meaningfully different final_pnl --
    confirming _run_sweep isn't silently resetting non-swept fields to
    SimConfig()'s defaults."""
    base_tight = _small_base(mm_half_spread=0.01)
    base_wide = _small_base(mm_half_spread=0.20)
    df_tight = sweep_hedge_frequency(base_tight, values=(1, 5), n_repeats=3)
    df_wide = sweep_hedge_frequency(base_wide, values=(1, 5), n_repeats=3)
    assert df_tight["mean_abs_inventory"].mean() != df_wide["mean_abs_inventory"].mean() \
        or df_tight["final_pnl"].mean() != df_wide["final_pnl"].mean()


# ============================================================
# 5. Each of the five sweep functions individually
# ============================================================

def test_sweep_hedge_frequency_varies_hedge_every():
    df = sweep_hedge_frequency(_small_base(), values=(1, 21), n_repeats=1)
    assert set(df["hedge_every"]) == {1, 21}


def test_sweep_spread_varies_mm_half_spread():
    df = sweep_spread(_small_base(), values=(0.02, 0.15), n_repeats=1)
    assert set(df["mm_half_spread"]) == {0.02, 0.15}


def test_sweep_volatility_mismatch_varies_realized_vol():
    df = sweep_volatility_mismatch(_small_base(), values=(0.15, 0.35), n_repeats=1)
    assert set(df["realized_vol"]) == {0.15, 0.35}


def test_sweep_volatility_mismatch_vol_mismatch_column_is_correct():
    base = _small_base(pricing_vol=0.25)
    df = sweep_volatility_mismatch(base, values=(0.15, 0.35), n_repeats=1)
    for _, row in df.iterrows():
        assert abs(row["vol_mismatch"] - (row["realized_vol"] - 0.25)) < 1e-12


def test_sweep_order_flow_varies_order_flow_intensity():
    df = sweep_order_flow(_small_base(), values=(0.5, 2.0), n_repeats=1)
    assert set(df["order_flow_intensity"]) == {0.5, 2.0}


def test_sweep_inventory_limits_varies_max_inventory():
    df = sweep_inventory_limits(_small_base(), values=(5, 50), n_repeats=1)
    assert set(df["max_inventory"]) == {5, 50}


# ============================================================
# 6. run_all_experiments
# ============================================================

def test_run_all_experiments_returns_five_expected_keys():
    results = run_all_experiments(_small_base(), n_repeats=1)
    expected_keys = {"hedge_every", "mm_half_spread", "realized_vol",
                      "order_flow_intensity", "max_inventory"}
    assert set(results.keys()) == expected_keys


def test_run_all_experiments_values_are_nonempty_dataframes():
    results = run_all_experiments(_small_base(), n_repeats=1)
    for key, df in results.items():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


def test_run_all_experiments_works_with_no_base_config():
    """base_config=None should default to SimConfig() without raising
    -- kept fast here via a monkeypatched-style small n_repeats, but
    still uses SimConfig()'s default n_steps=252, so this is the one
    slower test in the file (still just 5 sweeps x a few values x 1
    repeat)."""
    results = run_all_experiments(base_config=None, n_repeats=1)
    assert len(results) == 5
    for df in results.values():
        assert len(df) > 0


# ============================================================
# Plain-Python runner (works without pytest installed)
# ============================================================

def _all_test_functions():
    return [obj for name, obj in sorted(globals().items())
            if name.startswith("test_") and callable(obj)]


if __name__ == "__main__":
    tests = _all_test_functions()
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)