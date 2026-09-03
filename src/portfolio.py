"""
portfolio.py
============
Tracks cash, option inventory, and underlying shares, and decomposes
total P&L into option market-making P&L and delta-hedging P&L.

Two inventories, kept deliberately separate
--------------------------------------------
Portfolio.option_qty and MarketMaker.inventory (market_maker.py) are
two SEPARATE dictionaries, both keyed by OptionContract, updated in
lockstep whenever a fill happens -- not one shared object.
MarketMaker's copy drives quoting decisions (skew, caps) and belongs
to market-making logic; Portfolio's copy drives cash/P&L accounting
and belongs to accounting logic. simulation.py (a later module) is the
orchestrator responsible for calling both mm.execute(...) and
Portfolio.apply_option_fill(...) for the same fill, keeping the two in
sync. Deliberate separation of concerns, not accidental duplication.

Portfolio has no dependency on market_maker.py or hedging.py -- its
update methods take plain parameters (contract, side, price, qty,
commission), not Fill/HedgeTrade objects. It DOES depend on market.py
(OptionMarket), since valuing the option book and aggregating Greeks
requires theo_value()/greeks().

P&L decomposition, per timestep
--------------------------------
    option_pnl_step = d(option book market value) + option cash flow
    hedge_pnl_step  = d(shares market value)       + hedge cash flow
    total_pnl_step  = option_pnl_step + hedge_pnl_step

"Cash flow" is what was paid/received on trades executed since the
last PnLTracker.record() call. Every dollar of value change and every
dollar of cash flow lands in exactly one bucket, so the two always sum
to the true change in total portfolio value -- by construction.

Timing convention (read before calling record() / check_pnl_conservation()):
  - Apply ALL fills for a step (apply_option_fill / apply_hedge_trade)
    BEFORE calling record() for that step. record() compares "value
    now" to "value as of the last record() call"; a fill applied after
    record() but intended for that step will be misattributed to the
    next one.
  - record() must be called for EVERY step, in order, even if nothing
    traded that step (cash-flow deltas are just 0) -- it's also what
    resets the cash-flow accumulators, so skipping a call leaks that
    step's flow into the next.
  - check_pnl_conservation()'s total_value_series must be
    Portfolio.total_value(...) evaluated at that SAME point each step
    (after that step's fills -- same timing as record()).
    initial_value defaults to 0.0: the value of a fresh Portfolio()
    before any trade (zero cash, inventory, shares) -- NOT the first
    entry of total_value_series, which may already reflect step-0 fills.
"""

import numpy as np


class Portfolio:
    def __init__(self):
        self.cash = 0.0
        self.option_qty = {}  # contract -> signed qty held
        self.shares = 0.0

        # Cumulative cash flow since the last PnLTracker.record() call;
        # reset to zero by the tracker after each step is recorded.
        self.option_cash_flow = 0.0
        self.hedge_cash_flow = 0.0

    def apply_option_fill(self, contract, side: str, price: float, qty: int):
        """side is from the MM's perspective: 'buy' or 'sell'."""
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        signed = qty if side == "buy" else -qty
        flow = -price * qty if side == "buy" else price * qty
        self.option_qty[contract] = self.option_qty.get(contract, 0) + signed
        self.cash += flow
        self.option_cash_flow += flow

    def apply_hedge_trade(self, qty: float, price: float, commission: float = 0.0):
        """qty is signed: positive = bought shares, negative = sold."""
        flow = -qty * price - commission
        self.shares += qty
        self.cash += flow
        self.hedge_cash_flow += flow

    def option_market_value(self, market, S: float, t: float) -> float:
        total = 0.0
        for contract, qty in self.option_qty.items():
            if qty:
                total += qty * market.theo_value(contract, S, t)
        return total

    def net_option_delta(self, market, S: float, t: float) -> float:
        """Total delta of the option book alone (shares-equivalent),
        NOT including shares held. This is exactly the value to pass
        to DeltaHedger.hedge() as option_delta_exposure."""
        total = 0.0
        for contract, qty in self.option_qty.items():
            if qty:
                total += qty * market.greeks(contract, S, t).delta
        return total

    def net_option_gamma(self, market, S: float, t: float) -> float:
        """Risk reporting only -- nothing trades against this."""
        total = 0.0
        for contract, qty in self.option_qty.items():
            if qty:
                total += qty * market.greeks(contract, S, t).gamma
        return total

    def net_option_vega(self, market, S: float, t: float) -> float:
        """Risk reporting only -- nothing trades against this."""
        total = 0.0
        for contract, qty in self.option_qty.items():
            if qty:
                total += qty * market.greeks(contract, S, t).vega
        return total

    def total_value(self, market, S: float, t: float) -> float:
        return self.cash + self.option_market_value(market, S, t) + self.shares * S


class PnLTracker:
    """Records the per-step P&L decomposition described above. Call
    record() once per simulation step, in order, after that step's
    fills have been applied -- see the module docstring's timing
    convention."""

    def __init__(self):
        self.history = []  # list of dicts, one per timestep

    def record(self, t: float, S: float, portfolio: Portfolio, market,
               prev_option_mv: float, prev_shares_mv: float):
        option_mv = portfolio.option_market_value(market, S, t)
        shares_mv = portfolio.shares * S

        option_pnl_step = (option_mv - prev_option_mv) + portfolio.option_cash_flow
        hedge_pnl_step = (shares_mv - prev_shares_mv) + portfolio.hedge_cash_flow

        portfolio.option_cash_flow = 0.0
        portfolio.hedge_cash_flow = 0.0

        self.history.append(dict(
            t=t,
            S=S,
            option_pnl_step=option_pnl_step,
            hedge_pnl_step=hedge_pnl_step,
            total_pnl_step=option_pnl_step + hedge_pnl_step,
        ))
        return option_mv, shares_mv


def check_pnl_conservation(total_value_series, pnl_step_series,
                            initial_value: float = 0.0, tol: float = 1e-6) -> float:
    """Hard consistency check on the accounting convention above.

    total_value_series must be Portfolio.total_value(...) evaluated at
    the SAME point in each step as when PnLTracker.record() was called
    for that step (after that step's fills). initial_value is the
    portfolio's value BEFORE the first trade -- 0.0 for a fresh
    Portfolio() -- not the series' first entry.

    Asserts cumsum(pnl_step_series) == total_value_series - initial_value
    at every step, within `tol`. Raises AssertionError if violated --
    a hard requirement, not an optional diagnostic: a violation means
    the P&L decomposition has a bookkeeping bug and nothing downstream
    can be trusted. Returns the max absolute discrepancy observed.
    """
    cum_pnl = np.asarray(pnl_step_series.cumsum())
    realized_change = np.asarray(total_value_series) - initial_value
    max_abs_diff = float(np.abs(cum_pnl - realized_change).max())
    if max_abs_diff > tol:
        raise AssertionError(
            f"P&L conservation violated: max abs diff = {max_abs_diff:.6g} "
            f"(tolerance = {tol:.1e}).")
    return max_abs_diff