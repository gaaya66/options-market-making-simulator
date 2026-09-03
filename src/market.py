"""
market.py
=========
The underlying price process (geometric Brownian motion) and the
synthetic options market: a fixed basket of European option contracts
priced off Black-Scholes with a chosen "pricing volatility".

Realized vol vs. pricing vol -- kept strictly separate here:
  - realized_vol drives simulate_gbm() -- what the underlying actually
    does over the simulated path.
  - pricing_vol is the only volatility OptionMarket ever uses -- what
    every theo_value()/greeks() call is priced and hedged off.

Neither function in this module ever mixes the two. In reality a
market maker never knows future realized vol either -- only an
estimate (implied vol), which is exactly what pricing_vol represents.
The gap between the two is a primary source of an options desk's P&L,
and is deliberately left exposed as two independent parameters rather
than being collapsed into one "vol" input.
"""

import math
from dataclasses import dataclass

import numpy as np

from .pricing import bs_price, bs_greeks, Greeks


def simulate_gbm(S0, mu, sigma, dt, n_steps, seed=None):
    """Simulate one path of geometric Brownian motion:

        dS = mu * S * dt + sigma * S * dW

    Implemented via the exact log-Euler solution:

        S(t+dt) = S(t) * exp[(mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z]

    This is the exact distributional transition for GBM (not a Euler
    approximation of it), so each sampled step has no discretization
    bias regardless of how large dt is. That does NOT mean a coarse
    timestep captures everything about the continuous path, though:
    what happens *between* samples -- e.g. the path's extrema, or
    whether it crossed some level intrastep -- is simply not observed
    at all when dt is large. The exact-solution property only says the
    sampled points themselves are unbiased draws from the true GBM
    distribution; it says nothing about intrastep behavior, which this
    simulator never needs (nothing here is path-dependent within a
    step) but is worth being precise about.

    Returns an array of length n_steps + 1, with path[0] == S0.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_steps)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    path = np.empty(n_steps + 1)
    path[0] = S0
    path[1:] = S0 * np.exp(np.cumsum(log_returns))
    return path


@dataclass(frozen=True)
class OptionContract:
    """A single European option, identified by strike/maturity/type.
    Frozen + hashable so it can be used directly as a dict key for
    inventory tracking elsewhere in the simulator (market_maker.py,
    portfolio.py)."""

    strike: float
    maturity: float  # time to expiry in years, measured from t=0
    option_type: str  # 'call' or 'put'

    @property
    def name(self):
        return f"{self.option_type[0].upper()}{self.strike:g}"


class OptionMarket:
    """Prices a fixed basket of contracts against an evolving underlying.

    OptionMarket does no math of its own -- it only knows which values
    (pricing_vol, r, and time-to-expiry for a given contract/time) to
    feed into pricing.py's functions. All Black-Scholes formulas live
    in pricing.py; this class is purely an orchestration layer over it.
    """

    def __init__(self, contracts, r: float, pricing_vol: float):
        self.contracts = contracts
        self.r = r
        self.pricing_vol = pricing_vol

    def time_to_expiry(self, contract: OptionContract, t: float) -> float:
        """Time remaining, computed fresh from the contract's fixed
        maturity and the current simulation time. Clipped at 0 so a
        contract past its maturity always reports T = 0 exactly,
        matching pricing.py's exact (not epsilon-approximated) T <= 0
        branch."""
        return max(contract.maturity - t, 0.0)

    def theo_value(self, contract: OptionContract, S: float, t: float) -> float:
        """Theoretical (Black-Scholes) value at the current time.

        At T <= 0 this returns exact intrinsic value, via pricing.py's
        own T <= 0 branch -- T is passed through unmodified, never
        substituted with a small epsilon (see pricing.py for why that
        substitution is wrong).

        Important: this is a VALUATION only. Returning intrinsic value
        at expiry does not, by itself, close out or settle any option
        position -- it just tells you what the position is worth right
        now. Actually converting an expired position into a cash
        settlement (and removing it from inventory) is the portfolio's
        responsibility, done explicitly in portfolio.py, not an
        automatic side effect of calling this method.
        """
        T = self.time_to_expiry(contract, t)
        return bs_price(S, contract.strike, T, self.r, self.pricing_vol,
                         contract.option_type)

    def greeks(self, contract: OptionContract, S: float, t: float) -> Greeks:
        """Greeks at the current time, same T-passthrough reasoning as
        theo_value(). At expiry, degenerates to pricing.py's exact
        values (delta = 0/1 or 0/-1 by moneyness; gamma = vega = 0)."""
        T = self.time_to_expiry(contract, t)
        return bs_greeks(S, contract.strike, T, self.r, self.pricing_vol,
                          contract.option_type)