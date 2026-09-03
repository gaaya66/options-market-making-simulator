"""
tests/test_analytics.py
========================
Unit tests for src/analytics.py: sharpe_ratio, max_drawdown,
hedging_error, inventory_diagnostics, summarize_run, plot_simulation.

Run with:
    pytest tests/test_analytics.py
or directly:
    python tests/test_analytics.py

Scope: analytics.py is a pure, read-only consumer of the history/
pnl_history DataFrames run_simulation() produces. Most tests here use
small, hand-built DataFrames/Series with known-by-construction expected
outputs, rather than only checking properties of a full stochastic
run -- this makes the exact formulas independently verifiable. A
handful of tests do run a small, fixed-seed run_simulation() to check
summarize_run()/plot_simulation() against realistic data end-to-end.

Sections:
    1. sharpe_ratio
    2. max_drawdown
    3. hedging_error
    4. inventory_diagnostics
    5. summarize_run
    6. plot_simulation
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analytics import (  # noqa: E402
    sharpe_ratio, max_drawdown, hedging_error, inventory_diagnostics,
    summarize_run, plot_simulation,
)
from src.simulation import SimConfig, run_simulation  # noqa: E402


# ============================================================
# 1. sharpe_ratio
# ============================================================

def test_sharpe_ratio_matches_hand_computed_value():
    pnl_steps = [1.0, -1.0, 1.0, -1.0, 2.0]
    arr = np.asarray(pnl_steps)
    expected = (arr.mean() / arr.std()) * np.sqrt(252)
    assert abs(sharpe_ratio(pnl_steps) - expected) < 1e-9


def test_sharpe_ratio_zero_for_all_zero_pnl():
    assert sharpe_ratio([0.0, 0.0, 0.0, 0.0]) == 0.0


def test_sharpe_ratio_zero_for_constant_nonzero_pnl():
    """Constant nonzero P&L still has std == 0 -- should return 0.0,
    not inf or NaN."""
    result = sharpe_ratio([5.0, 5.0, 5.0, 5.0])
    assert result == 0.0
    assert not np.isnan(result)
    assert not np.isinf(result)


def test_sharpe_ratio_scales_with_sqrt_periods_per_year():
    pnl_steps = [1.0, -0.5, 2.0, -1.5, 0.5]
    s_252 = sharpe_ratio(pnl_steps, periods_per_year=252)
    s_504 = sharpe_ratio(pnl_steps, periods_per_year=504)
    assert abs(s_504 - s_252 * np.sqrt(2.0)) < 1e-9


def test_sharpe_ratio_negative_for_negative_mean_pnl():
    pnl_steps = [-1.0, -2.0, 0.5, -1.5, -0.5]
    assert sharpe_ratio(pnl_steps) < 0


def test_sharpe_ratio_accepts_a_pandas_series():
    s = pd.Series([1.0, -1.0, 2.0, -2.0, 0.5])
    result = sharpe_ratio(s)
    expected = (s.to_numpy().mean() / s.to_numpy().std()) * np.sqrt(252)
    assert abs(result - expected) < 1e-9


# ============================================================
# 2. max_drawdown
# ============================================================

def test_max_drawdown_zero_for_monotonically_increasing_series():
    cum_pnl = [0, 5, 10, 15, 25]
    assert max_drawdown(cum_pnl) == 0.0


def test_max_drawdown_matches_hand_computed_worst_decline():
    # Peaks at 15 (index 3), falls to 2 (index 4) -> drawdown = 2 - 15 = -13
    cum_pnl = [0, 10, 5, 15, 2]
    assert max_drawdown(cum_pnl) == -13


def test_max_drawdown_all_decreasing_equals_final_minus_first():
    cum_pnl = [20, 15, 10, 5, 0]
    assert max_drawdown(cum_pnl) == 0 - 20


def test_max_drawdown_single_point_is_zero():
    assert max_drawdown([42.0]) == 0.0


def test_max_drawdown_picks_the_worst_of_multiple_declines():
    # First decline: 10 -> 8 (-2). Second, worse decline: 20 -> 5 (-15).
    cum_pnl = [0, 10, 8, 20, 5, 12]
    assert max_drawdown(cum_pnl) == -15


# ============================================================
# 3. hedging_error
# ============================================================

def test_hedging_error_zero_when_net_delta_always_zero():
    history = pd.DataFrame({"net_delta": [0.0, 0.0, 0.0]})
    assert hedging_error(history) == 0.0


def test_hedging_error_matches_hand_computed_rms():
    history = pd.DataFrame({"net_delta": [1.0, -1.0, 2.0, -2.0]})
    # mean of squares = (1 + 1 + 4 + 4) / 4 = 2.5 -> sqrt(2.5)
    assert abs(hedging_error(history) - np.sqrt(2.5)) < 1e-9


def test_hedging_error_ignores_other_columns():
    minimal = pd.DataFrame({"net_delta": [1.0, -1.0, 2.0, -2.0]})
    extra = pd.DataFrame({
        "net_delta": [1.0, -1.0, 2.0, -2.0],
        "S": [100.0, 101.0, 99.0, 98.0],
        "shares": [5, 6, 7, 8],
    })
    assert hedging_error(minimal) == hedging_error(extra)


# ============================================================
# 4. inventory_diagnostics
# ============================================================

def _diagnostics_history():
    return pd.DataFrame({
        "total_abs_inventory": [0, 3, 5, 2],
        "hedge_qty": [0.0, 1.5, 0.0, -0.5],
        "option_gamma": [0.0, 0.1, -0.2, 0.3],
        "option_vega": [0.0, 10.0, -5.0, 2.0],
    })


def test_inventory_diagnostics_max_and_mean_abs_inventory():
    diag = inventory_diagnostics(_diagnostics_history())
    assert diag["max_abs_inventory"] == 5.0
    assert diag["mean_abs_inventory"] == (0 + 3 + 5 + 2) / 4


def test_inventory_diagnostics_counts_only_nonzero_hedge_trades():
    diag = inventory_diagnostics(_diagnostics_history())
    assert diag["n_hedge_trades"] == 2  # 1.5 and -0.5 are nonzero


def test_inventory_diagnostics_mean_abs_gamma_and_vega():
    diag = inventory_diagnostics(_diagnostics_history())
    assert abs(diag["mean_abs_gamma_exposure"] - (0 + 0.1 + 0.2 + 0.3) / 4) < 1e-9
    assert abs(diag["mean_abs_vega_exposure"] - (0 + 10 + 5 + 2) / 4) < 1e-9


def test_inventory_diagnostics_all_zero_gamma_vega_gives_zero_means():
    history = pd.DataFrame({
        "total_abs_inventory": [1, 2],
        "hedge_qty": [0.0, 0.0],
        "option_gamma": [0.0, 0.0],
        "option_vega": [0.0, 0.0],
    })
    diag = inventory_diagnostics(history)
    assert diag["mean_abs_gamma_exposure"] == 0.0
    assert diag["mean_abs_vega_exposure"] == 0.0
    assert diag["n_hedge_trades"] == 0


def test_inventory_diagnostics_returns_exactly_five_keys():
    diag = inventory_diagnostics(_diagnostics_history())
    expected_keys = {"max_abs_inventory", "mean_abs_inventory", "n_hedge_trades",
                      "mean_abs_gamma_exposure", "mean_abs_vega_exposure"}
    assert set(diag.keys()) == expected_keys


# ============================================================
# 5. summarize_run
# ============================================================

def _small_run():
    cfg = SimConfig(seed=0, n_steps=60)
    return run_simulation(cfg)


def test_summarize_run_contains_all_expected_keys():
    history, pnl_history = _small_run()
    summary = summarize_run(history, pnl_history)
    expected_keys = {
        "final_pnl", "final_option_pnl", "final_hedge_pnl", "sharpe",
        "max_drawdown", "hedging_error", "max_abs_inventory",
        "mean_abs_inventory", "n_hedge_trades", "mean_abs_gamma_exposure",
        "mean_abs_vega_exposure",
    }
    assert expected_keys.issubset(set(summary.keys()))


def test_summarize_run_values_are_all_finite():
    history, pnl_history = _small_run()
    summary = summarize_run(history, pnl_history)
    for key, value in summary.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"


def test_summarize_run_final_pnl_matches_manual_cumsum():
    history, pnl_history = _small_run()
    summary = summarize_run(history, pnl_history)
    expected = pnl_history["total_pnl_step"].cumsum().iloc[-1]
    assert abs(summary["final_pnl"] - expected) < 1e-9


def test_summarize_run_sharpe_matches_direct_call():
    history, pnl_history = _small_run()
    summary = summarize_run(history, pnl_history)
    expected = sharpe_ratio(pnl_history["total_pnl_step"])
    assert summary["sharpe"] == expected


def test_summarize_run_max_drawdown_matches_direct_call():
    history, pnl_history = _small_run()
    summary = summarize_run(history, pnl_history)
    expected = max_drawdown(pnl_history["total_pnl_step"].cumsum())
    assert summary["max_drawdown"] == expected


def test_summarize_run_hedging_error_matches_direct_call():
    history, pnl_history = _small_run()
    summary = summarize_run(history, pnl_history)
    expected = hedging_error(history)
    assert summary["hedging_error"] == expected


def test_summarize_run_inventory_fields_match_direct_call():
    history, pnl_history = _small_run()
    summary = summarize_run(history, pnl_history)
    expected = inventory_diagnostics(history)
    for key, value in expected.items():
        assert summary[key] == value


# ============================================================
# 6. plot_simulation
# ============================================================

def test_plot_simulation_returns_a_figure_with_four_axes():
    history, pnl_history = _small_run()
    fig = plot_simulation(history, pnl_history)
    # 4 primary axes + 1 twinx() secondary axis on panel 2 = 5 Axes objects
    assert len(fig.axes) == 5
    plt.close(fig)


def test_plot_simulation_saves_a_nonempty_file(tmp_path=None):
    history, pnl_history = _small_run()
    out_path = os.path.join(os.path.dirname(__file__), "_test_plot_output.png")
    try:
        fig = plot_simulation(history, pnl_history, save_path=out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
        plt.close(fig)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_plot_simulation_without_save_path_writes_no_file():
    history, pnl_history = _small_run()
    before = set(os.listdir(os.path.dirname(__file__)))
    fig = plot_simulation(history, pnl_history)
    after = set(os.listdir(os.path.dirname(__file__)))
    assert before == after
    plt.close(fig)


def test_plot_simulation_does_not_mutate_input_dataframes():
    history, pnl_history = _small_run()
    history_before = history.copy(deep=True)
    pnl_history_before = pnl_history.copy(deep=True)
    fig = plot_simulation(history, pnl_history)
    pd.testing.assert_frame_equal(history, history_before)
    pd.testing.assert_frame_equal(pnl_history, pnl_history_before)
    plt.close(fig)


def test_plot_simulation_accepts_custom_title():
    history, pnl_history = _small_run()
    fig = plot_simulation(history, pnl_history, title="Custom Title")
    assert fig.axes[0].get_title() == "Custom Title"
    plt.close(fig)


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