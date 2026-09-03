"""
simulation.py
=============
Orchestrates OptionMarket, MarketMaker, OrderFlowGenerator, Portfolio,
and DeltaHedger into a single timestepped simulation. This module
contains NO pricing, quoting, hedging, or P&L formulas of its own --
it only calls the five existing components in the right order and
records what happened.

The full underlying path is simulated ONCE upfront (simulate_gbm
already returns the whole path in one call); "the underlying moves"
inside the loop just means indexing into that array.

Per-step order (Steps A-F), and why it matters:
  A. Read this step's S, t from the pre-simulated path.
  B. Customer order flow fills against the MM's quotes. Skipped on the
     terminal step (see below). MarketMaker.execute() already clips
     the requested qty to the inventory cap; Portfolio.apply_option_fill
     is called with the FILL's actual executed qty (fill.qty), never
     the originally requested qty -- using the wrong one would
     silently desynchronize MarketMaker.inventory from
     Portfolio.option_qty, the two mirrored inventories from
     portfolio.py's design.
  C. Compute the option book's net delta from the portfolio AFTER all
     of this step's fills (not before) -- hedging against a stale,
     pre-fill delta would defeat the point of hedging the position
     that actually exists right now.
  D. Hedge, if this is a hedging step (DeltaHedger.should_hedge).
  E. Record P&L (PnLTracker.record), AFTER both B and D -- matching
     portfolio.py's documented timing convention ("apply all fills
     before calling record()"), which is what makes
     check_pnl_conservation meaningful.
  F. Build this step's history row. net_delta (option delta + shares)
     uses portfolio.shares AFTER step D, so it reads ~0 immediately
     after a hedge and drifts (via gamma) between hedges.

Terminal-step order flow: skipped. At i == n_steps, t normally equals
the contracts' maturity -- generating fresh flow into options at the
exact instant they settle to intrinsic value is a degenerate edge case
that adds a special branch without adding insight. Hedging and P&L
recording still happen normally on this step.

Expiry settlement: deliberately NOT implemented. Per market.py's
design, theo_value() at T<=0 is a VALUATION (exact intrinsic value),
not a settlement action. This module never converts an expiring
position into cash/shares or removes it from portfolio.option_qty --
if n_steps runs past a contract's maturity, that position simply
continues to mark at (now-fixed) intrinsic value. No exercise/
assignment mechanics are modeled.
"""

from dataclasses import dataclass

import pandas as pd

from .market import simulate_gbm, OptionContract, OptionMarket
from .market_maker import MarketMaker, OrderFlowGenerator
from .hedging import DeltaHedger
from .portfolio import Portfolio, PnLTracker, check_pnl_conservation


@dataclass
class SimConfig:
    # Underlying process (-> simulate_gbm)
    S0: float = 100.0
    mu: float = 0.05
    realized_vol: float = 0.25
    dt: float = 1 / 252
    n_steps: int = 252
    seed: int = None

    # Option market (-> OptionMarket)
    r: float = 0.02
    pricing_vol: float = 0.25
    strikes: tuple = (90, 100, 110)
    maturity: float = 0.5

    # Market maker (-> MarketMaker)
    mm_half_spread: float = 0.05
    inventory_skew: float = 0.002
    max_inventory: int = 50
    quote_size: int = 1

    # Order flow (-> OrderFlowGenerator)
    order_flow_intensity: float = 1.0
    buy_prob: float = 0.5
    order_size: int = 1

    # Hedging (-> DeltaHedger)
    hedge_every: int = 1
    hedge_spread_bps: float = 5.0
    hedge_commission: float = 0.0


def build_contracts(strikes, maturity):
    """One call and one put per strike -- the only place contract
    construction happens in this project."""
    contracts = []
    for k in strikes:
        contracts.append(OptionContract(strike=k, maturity=maturity, option_type="call"))
        contracts.append(OptionContract(strike=k, maturity=maturity, option_type="put"))
    return contracts


def run_simulation(config: SimConfig):
    """Runs one full simulation and returns (history, pnl_history),
    both pandas DataFrames indexed by simulation step."""

    path = simulate_gbm(config.S0, config.mu, config.realized_vol,
                         config.dt, config.n_steps, seed=config.seed)

    contracts = build_contracts(config.strikes, config.maturity)
    market = OptionMarket(contracts, config.r, config.pricing_vol)

    mm = MarketMaker(config.mm_half_spread, config.inventory_skew,
                      config.max_inventory, config.quote_size)
    flow_seed = None if config.seed is None else config.seed + 1
    flow = OrderFlowGenerator(config.order_flow_intensity, config.buy_prob,
                               config.order_size, seed=flow_seed)
    hedger = DeltaHedger(config.hedge_every, config.hedge_spread_bps,
                          config.hedge_commission)

    portfolio = Portfolio()
    pnl_tracker = PnLTracker()

    prev_option_mv, prev_shares_mv = 0.0, 0.0
    rows = []

    for i in range(config.n_steps + 1):
        # --- A: read this step's price off the pre-simulated path ---
        t = i * config.dt
        S = path[i]
        hedge_qty = 0.0

        # --- B: customer order flow fills against the MM's quotes ---
        if i < config.n_steps:  # no new flow on the terminal step
            for contract, mm_side, qty in flow.generate(contracts):
                theo = market.theo_value(contract, S, t)
                quote = mm.quote(contract, theo)
                price = quote.ask if mm_side == "sell" else quote.bid
                fill = mm.execute(t, contract, mm_side, price, qty)
                if fill.qty > 0:  # use the ACTUAL clipped fill quantity
                    portfolio.apply_option_fill(contract, mm_side, price, fill.qty)

        # --- C: option book delta, computed AFTER this step's fills ---
        option_delta = portfolio.net_option_delta(market, S, t)

        # --- D: hedge, if this is a hedging step ---
        if hedger.should_hedge(i):
            trade = hedger.hedge(t, S, option_delta, portfolio.shares)
            if trade.qty != 0:
                portfolio.apply_hedge_trade(trade.qty, trade.price, trade.commission)
                hedge_qty = trade.qty

        # --- E: record P&L, AFTER both fills (B) and hedging (D) ---
        prev_option_mv, prev_shares_mv = pnl_tracker.record(
            t, S, portfolio, market, prev_option_mv, prev_shares_mv)

        # --- F: risk aggregates and this step's history row ---
        option_gamma = portfolio.net_option_gamma(market, S, t)
        option_vega = portfolio.net_option_vega(market, S, t)
        net_delta = option_delta + portfolio.shares  # post-hedge (D)
        # Gross inventory SUMMED ACROSS ALL CONTRACTS in the book --
        # NOT a per-contract figure, and not bounded by max_inventory
        # alone (max_inventory caps each contract independently; with
        # N contracts, this sum can legitimately reach
        # max_inventory * N). See analytics.inventory_diagnostics for
        # the same note at the point this is consumed.
        total_abs_inventory = sum(abs(q) for q in portfolio.option_qty.values())

        rows.append(dict(
            step=i, t=t, S=S,
            total_abs_inventory=total_abs_inventory,
            shares=portfolio.shares,
            option_delta=option_delta,
            option_gamma=option_gamma,
            option_vega=option_vega,
            net_delta=net_delta,
            hedge_qty=hedge_qty,
            total_value=portfolio.total_value(market, S, t),
        ))

    history = pd.DataFrame(rows)
    pnl_history = pd.DataFrame(pnl_tracker.history)

    # Hard requirement, not an optional diagnostic: let this raise if
    # it fails -- a violation means the returned DataFrames can't be
    # trusted downstream.
    check_pnl_conservation(history["total_value"], pnl_history["total_pnl_step"])

    return history, pnl_history