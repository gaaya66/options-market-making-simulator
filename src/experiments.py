"""
experiments.py
==============
Five parameter sweeps investigating how the market maker's P&L and
risk profile change with hedge frequency, quote width, volatility
mismatch, order-flow intensity, and inventory limits.

Each sweep varies exactly one SimConfig field, holding every other
field fixed at the base config, runs run_simulation() once per
(value, repeat) pair, calls analytics.summarize_run() on the result,
and returns the RAW per-run results as a pandas DataFrame -- one row
per (value, repeat). No aggregation (mean/std across repeats) and no
plotting happen in this module; both are left to the caller
(analytics.py owns plotting; groupby/agg on the returned DataFrame is
the caller's job if wanted).

Common random numbers, not parameter tuning
---------------------------------------------
Within a sweep, repeat `rep` always uses seed=rep, REGARDLESS of which
parameter value it's paired with. This is deliberate: it means every
swept value at repeat 0 is simulated against the exact same underlying
price path and the exact same customer-order-flow draws as every other
swept value at repeat 0 (since simulate_gbm and OrderFlowGenerator are
both seeded from that same `rep`). This is the "common random numbers"
variance-reduction technique -- it isolates the EFFECT OF THE SWEPT
PARAMETER from the effect of which random path happened to be drawn,
so a difference between two parameter values is more likely to reflect
the parameter itself rather than random noise between unrelated paths.

This is NOT parameter tuning or optimization. No function in this
module selects a "best" value, compares configurations to pick a
winner, or fits a parameter to any objective. Every sweep returns the
full, raw distribution across all values and repeats -- including
configurations where the market maker performs poorly -- because that
is the point of a sensitivity study: describing how outcomes change
with a parameter, not searching for a favorable one. See the project
README for the same framing applied to how results should be read.

Exactly five sweeps, no more:
    sweep_hedge_frequency      -- hedge_every
    sweep_spread                -- mm_half_spread
    sweep_volatility_mismatch    -- realized_vol (pricing_vol held fixed)
    sweep_order_flow               -- order_flow_intensity
    sweep_inventory_limits           -- max_inventory
"""

import copy

import pandas as pd

from .simulation import SimConfig, run_simulation
from .analytics import summarize_run


def _run_sweep(base_config: SimConfig, param_name: str, values, n_repeats: int = 5) -> pd.DataFrame:
    """Generic sweep engine used by all five experiments below.

    For each value in `values`, for each rep in range(n_repeats):
    deep-copies base_config (never mutates the caller's object), sets
    cfg.<param_name> = value, sets cfg.seed = rep (common random
    numbers -- see module docstring), runs the simulation, and calls
    summarize_run() on the result. Returns one row per (value, rep)
    pair, with columns = every summarize_run() key, plus param_name
    and rep.
    """
    rows = []
    for value in values:
        for rep in range(n_repeats):
            cfg = copy.deepcopy(base_config)
            setattr(cfg, param_name, value)
            cfg.seed = rep
            history, pnl_history = run_simulation(cfg)
            summary = summarize_run(history, pnl_history)
            summary[param_name] = value
            summary["rep"] = rep
            rows.append(summary)
    return pd.DataFrame(rows)


def sweep_hedge_frequency(base_config: SimConfig, values=(1, 2, 5, 10, 21),
                           n_repeats: int = 5) -> pd.DataFrame:
    """Hedge every 1, 2, 5, 10, 21 steps (SimConfig.hedge_every).

    Question: does hedging more often reduce risk (drawdown, hedging
    error) faster than it costs in trading expense -- and does mean
    P&L actually change, or does the cost of infrequent hedging show
    up mainly as VARIANCE rather than mean?
    """
    return _run_sweep(base_config, "hedge_every", values, n_repeats)


def sweep_spread(base_config: SimConfig, values=(0.01, 0.02, 0.05, 0.10, 0.15),
                  n_repeats: int = 5) -> pd.DataFrame:
    """Market maker's own half-spread over theoretical value
    (SimConfig.mm_half_spread).

    Question: how much does quote width affect P&L per unit of flow,
    holding flow volume fixed?

    Honest framing: order flow in this simulator (OrderFlowGenerator)
    is an exogenous Poisson process that does NOT respond to quote
    width -- widening the spread never reduces how much flow arrives,
    only how much edge is captured per fill. This sweep therefore
    shows "value of edge per fill, holding volume fixed," NOT the
    realistic trade-off where wider spreads also reduce fill
    probability/volume. That trade-off would require a price-sensitive
    flow model, which is out of scope for this project.
    """
    return _run_sweep(base_config, "mm_half_spread", values, n_repeats)


def sweep_volatility_mismatch(base_config: SimConfig,
                               values=(0.10, 0.15, 0.25, 0.35, 0.50),
                               n_repeats: int = 5) -> pd.DataFrame:
    """Realized volatility of the underlying (SimConfig.realized_vol),
    while pricing_vol stays fixed at base_config.pricing_vol throughout
    every run in the sweep.

    Question: the central one for this project -- what happens to a
    delta-hedged option book's P&L when the volatility it was priced
    with turns out to be wrong? Values of realized_vol below
    pricing_vol mean the book was priced "rich" relative to what
    actually happened; values above mean "cheap".

    Adds a derived `vol_mismatch` column (realized_vol - pricing_vol)
    for readability, so a reader doesn't need to separately look up
    base_config.pricing_vol to see how far each row is from correctly
    priced.
    """
    df = _run_sweep(base_config, "realized_vol", values, n_repeats)
    df["vol_mismatch"] = df["realized_vol"] - base_config.pricing_vol
    return df


def sweep_order_flow(base_config: SimConfig, values=(0.25, 0.5, 1.0, 2.0, 4.0),
                      n_repeats: int = 5) -> pd.DataFrame:
    """Expected customer orders per contract per timestep
    (SimConfig.order_flow_intensity).

    Question: how much does the volume of customer flow matter,
    holding edge and hedging cadence fixed -- does P&L scale roughly
    linearly with flow, or does something else (e.g. inventory caps
    binding more often) change that relationship?
    """
    return _run_sweep(base_config, "order_flow_intensity", values, n_repeats)


def sweep_inventory_limits(base_config: SimConfig, values=(5, 10, 25, 50, 100),
                            n_repeats: int = 5) -> pd.DataFrame:
    """Per-contract inventory cap (SimConfig.max_inventory).

    Question: what's the cost, in lost flow/edge, of capping risk
    tightly -- and does a tight cap actually reduce drawdown/hedging
    error, or does it just cap gross inventory without meaningfully
    changing the risk metrics?
    """
    return _run_sweep(base_config, "max_inventory", values, n_repeats)


SWEEPS = [
    ("hedge_every", sweep_hedge_frequency),
    ("mm_half_spread", sweep_spread),
    ("realized_vol", sweep_volatility_mismatch),
    ("order_flow_intensity", sweep_order_flow),
    ("max_inventory", sweep_inventory_limits),
]


def run_all_experiments(base_config: SimConfig = None, n_repeats: int = 5) -> dict:
    """Runs all five sweeps and returns {param_name: DataFrame}, keyed
    by the SimConfig field name (not the function name) so a caller
    can index e.g. results["mm_half_spread"] directly. base_config
    defaults to SimConfig() if not given."""
    base_config = base_config or SimConfig()
    return {param: sweep_fn(base_config, n_repeats=n_repeats)
            for param, sweep_fn in SWEEPS}