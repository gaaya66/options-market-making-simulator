"""
hedging.py
==========
Discrete delta hedging: at a configurable cadence, trade the underlying
so that net portfolio delta (option book delta + shares currently
held) is brought back to exactly zero.

    Option position -> delta -> net portfolio delta -> trade underlying
    -> reduced delta exposure -> hedge commission recorded

Deliberately decoupled from pricing.py/market.py: this module never
computes a Greek itself and never sees an OptionContract. It receives
`option_delta_exposure` as a single, already-summed float -- the net
delta of the entire option book -- computed upstream (by portfolio.py).
That keeps this module trivially testable with made-up numbers, and
keeps "who owns computing delta" unambiguous.

Hedging error (how far net delta actually drifted between hedges,
driven by gamma) deliberately does NOT live here -- it needs the full
per-step history of net delta, which only exists once simulation.py is
looping over every step. This module only ever sees one instant at a
time, and only tracks hedge COSTS (via HedgeTrade.commission), not the
drift metric itself.

Simplifications: full re-hedge to zero every interval (no partial/
banded adjustment), delta-only (no gamma/vega hedging), unlimited
underlying liquidity (no position caps, no market impact), and a
constant spread (not size-dependent).
"""

from dataclasses import dataclass


@dataclass
class HedgeTrade:
    t: float
    qty: float          # shares bought (+) or sold (-); 0.0 if already hedged
    price: float         # execution price, spread already applied
    commission: float     # flat per-share trading cost, kept separate
                            # from the spread cost (which is embedded in
                            # `price` vs. the underlying mid S)


class DeltaHedger:
    def __init__(self, hedge_every: int, spread_bps: float,
                 commission_per_share: float = 0.0):
        """
        hedge_every:           hedge every N simulation steps
        spread_bps:             underlying half-spread, in basis points
                                 of the underlying price
        commission_per_share:   flat commission charged per share traded
        """
        self.hedge_every = hedge_every
        self.spread_bps = spread_bps
        self.commission_per_share = commission_per_share

    def should_hedge(self, step_index: int) -> bool:
        """Hedges fire at steps 0, hedge_every, 2*hedge_every, ... --
        including step 0, so the book starts hedged immediately."""
        return step_index % self.hedge_every == 0

    def hedge(self, t: float, S: float, option_delta_exposure: float,
              shares_held: float) -> HedgeTrade:
        """Compute and 'execute' the trade needed to bring net delta
        (option_delta_exposure + shares held after the trade) to zero.

        This is a FULL re-hedge every call, not a partial adjustment:
            target_shares = -option_delta_exposure
            trade_qty     = target_shares - shares_held
        """
        target_shares = -option_delta_exposure
        qty = target_shares - shares_held

        if qty == 0:
            return HedgeTrade(t=t, qty=0.0, price=S, commission=0.0)

        half_spread = S * self.spread_bps / 10_000.0
        exec_price = S + half_spread if qty > 0 else S - half_spread
        commission = abs(qty) * self.commission_per_share
        return HedgeTrade(t=t, qty=qty, price=exec_price, commission=commission)