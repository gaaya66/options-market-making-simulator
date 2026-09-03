"""
analytics.py
============
Risk metrics and plotting for a single simulation run. Pure consumer
of the two DataFrames run_simulation() already produces (history,
pnl_history) -- this module calls none of the five simulation
components directly, and never mutates its inputs.

Six functions: four scalar/dict metrics (sharpe_ratio, max_drawdown,
hedging_error, inventory_diagnostics), one bundler (summarize_run) that
experiments.py will call once per run inside each parameter sweep, and
one diagnostic plot (plot_simulation). No new trading logic and no new
metrics beyond this list -- analytics.py only describes what a run did,
it never influences it.

Note on sharpe_ratio: this is an ANNUALIZED P&L-BASED SHARPE METRIC,
not a conventional return-based Sharpe ratio. A textbook Sharpe ratio
is computed on PERIOD RETURNS (e.g. daily % return on invested
capital) relative to a risk-free rate. Here, pnl_steps is raw dollar
P&L per step -- there is no "capital base" to divide by (this
simulator tracks absolute portfolio value, not a return series), and
no risk-free rate is subtracted (the P&L is already excess of
financing costs, since portfolio.py models no interest on cash). The
formula (mean/std, annualized by sqrt(periods_per_year)) is the same
shape as a Sharpe ratio and answers the same kind of question --
"was this P&L worth its variance" -- but the units are dollars of P&L
per unit of P&L volatility, not returns per unit of return volatility.
Treat it as a comparable-across-runs risk-adjusted P&L metric, not as
a Sharpe ratio you could compare to a fund's published return Sharpe.
"""

import numpy as np
import matplotlib.pyplot as plt


def sharpe_ratio(pnl_steps, periods_per_year: int = 252) -> float:
    """Annualized P&L-based Sharpe metric (see module docstring for why
    this is not a conventional return Sharpe ratio):

        sharpe = (mean(pnl_steps) / std(pnl_steps)) * sqrt(periods_per_year)

    Returns 0.0 (rather than raising or returning inf/NaN) if
    std(pnl_steps) == 0 -- a degenerate, no-variance run.
    """
    pnl_steps = np.asarray(pnl_steps)
    std = pnl_steps.std()
    if std == 0:
        return 0.0
    return float((pnl_steps.mean() / std) * np.sqrt(periods_per_year))


def max_drawdown(cum_pnl) -> float:
    """Largest peak-to-trough decline in CUMULATIVE P&L (the caller
    passes an already-cumsum'd series, e.g.
    pnl_history["total_pnl_step"].cumsum()).

        running_max = cummax(cum_pnl)
        drawdown    = cum_pnl - running_max
        max_drawdown = min(drawdown)

    Returns a value <= 0 (or exactly 0.0 if cum_pnl never fell below a
    prior peak).
    """
    cum_pnl = np.asarray(cum_pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    return float(drawdown.min())


def hedging_error(history) -> float:
    """RMS (root-mean-square) of net delta across every recorded step:

        hedging_error = sqrt(mean(net_delta ** 2))

    Reads only history["net_delta"]. This is ~0 when hedging every
    step (net delta never has time to drift) and grows as hedging
    becomes less frequent, since gamma lets delta wander between
    hedges -- see simulation.py's Step F and its tests.
    """
    net_delta = np.asarray(history["net_delta"])
    return float(np.sqrt(np.mean(net_delta ** 2)))


def inventory_diagnostics(history) -> dict:
    """How much risk was carried, and how much trading it took to
    manage it. Reads total_abs_inventory, hedge_qty, option_gamma,
    option_vega from history.

    IMPORTANT: max_abs_inventory / mean_abs_inventory are GROSS,
    PORTFOLIO-WIDE figures -- total_abs_inventory (from simulation.py)
    is sum(abs(qty) for every contract in the book), summed ACROSS ALL
    CONTRACTS, not a single contract's position. This is a different
    quantity from SimConfig.max_inventory, which is a PER-CONTRACT cap
    enforced independently for each contract by MarketMaker.quote()/
    execute() (see market_maker.py). With N contracts in the book, a
    fully-loaded, independently-capped position in every contract can
    legitimately produce total_abs_inventory as high as
    max_inventory * N -- this is expected arithmetic, not a cap
    breach. To check whether any INDIVIDUAL contract ever exceeded its
    own cap, inspect MarketMaker.inventory directly (per contract);
    total_abs_inventory cannot answer that question by itself.

    Gamma/vega means matter specifically because nothing in this
    project ever trades against them (hedging.py/portfolio.py are
    delta-only by design) -- these numbers are the only visibility
    into how much of that un-hedged risk the book carried.
    """
    return dict(
        max_abs_inventory=float(history["total_abs_inventory"].max()),
        mean_abs_inventory=float(history["total_abs_inventory"].mean()),
        n_hedge_trades=int((history["hedge_qty"] != 0).sum()),
        mean_abs_gamma_exposure=float(history["option_gamma"].abs().mean()),
        mean_abs_vega_exposure=float(history["option_vega"].abs().mean()),
    )


def summarize_run(history, pnl_history) -> dict:
    """One flat dict of headline stats for a single simulation run --
    the single function experiments.py calls once per run inside each
    parameter sweep. Merges final P&L figures, the three risk metrics
    above, and inventory_diagnostics(history) into one dict."""
    cum_total = pnl_history["total_pnl_step"].cumsum()
    cum_option = pnl_history["option_pnl_step"].cumsum()
    cum_hedge = pnl_history["hedge_pnl_step"].cumsum()

    summary = dict(
        final_pnl=float(cum_total.iloc[-1]),
        final_option_pnl=float(cum_option.iloc[-1]),
        final_hedge_pnl=float(cum_hedge.iloc[-1]),
        sharpe=sharpe_ratio(pnl_history["total_pnl_step"]),
        max_drawdown=max_drawdown(cum_total),
        hedging_error=hedging_error(history),
    )
    summary.update(inventory_diagnostics(history))
    return summary


def plot_simulation(history, pnl_history, title: str = "Market-Making Simulation",
                     save_path: str = None):
    """Four-panel diagnostic plot for a single run, sharing a time axis:
      1. Underlying price
      2. Gross option inventory, with individual hedge trades overlaid
      3. Delta exposure: option_delta (pre-hedge) vs. net_delta (post-hedge)
      4. Cumulative P&L: total / option-MM / hedging, each cumsum'd

    Returns the Figure. If save_path is given, also saves it there
    (dpi=120). Does not mutate history or pnl_history.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(history["t"], history["S"], color="black", linewidth=1)
    axes[0].set_ylabel("Underlying price")
    axes[0].set_title(title)

    ax_inv = axes[1]
    ax_inv.plot(history["t"], history["total_abs_inventory"], color="tab:blue")
    ax_inv.set_ylabel("Gross option\ninventory", color="tab:blue")
    ax_hedge = ax_inv.twinx()
    hedge_mask = history["hedge_qty"] != 0
    ax_hedge.scatter(history.loc[hedge_mask, "t"], history.loc[hedge_mask, "hedge_qty"],
                      color="tab:orange", s=10)
    ax_hedge.axhline(0, color="grey", linewidth=0.6)
    ax_hedge.set_ylabel("Hedge trade\nsize (shares)", color="tab:orange")

    axes[2].plot(history["t"], history["option_delta"], label="Option delta (pre-hedge)")
    axes[2].plot(history["t"], history["net_delta"], label="Net delta (post-hedge)")
    axes[2].axhline(0, color="grey", linewidth=0.8)
    axes[2].set_ylabel("Delta exposure\n(shares)")
    axes[2].legend(loc="upper right", fontsize=8)

    cum_total = pnl_history["total_pnl_step"].cumsum()
    cum_option = pnl_history["option_pnl_step"].cumsum()
    cum_hedge = pnl_history["hedge_pnl_step"].cumsum()
    axes[3].plot(pnl_history["t"], cum_total, label="Total P&L", color="black")
    axes[3].plot(pnl_history["t"], cum_option, label="Option MM P&L", linestyle="--")
    axes[3].plot(pnl_history["t"], cum_hedge, label="Hedging P&L", linestyle="--")
    axes[3].axhline(0, color="grey", linewidth=0.8)
    axes[3].set_ylabel("Cumulative P&L")
    axes[3].set_xlabel("Time (years)")
    axes[3].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig