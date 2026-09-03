"""
pricing.py
==========
Black-Scholes pricing and Greeks for European options.

Assumptions (standard Black-Scholes / Akuna "Options 101" framing):
  - European exercise only (no early exercise).
  - No dividends on the underlying.
  - Constant, known risk-free rate and volatility over the life of
    the option (no smile, no term structure).
  - Continuous trading, no transaction costs at the pricing level
    (costs are added separately, at the market-maker/hedging level
    in later modules).

Units:
  - sigma is decimal (0.25 = 25% annualized volatility).
  - delta: d(price)/d(S), unitless (shares-equivalent per contract).
  - gamma: d(delta)/d(S).
  - vega:  d(price)/d(sigma), per 1.00 (100%) change in vol.
  - theta: standard trading-desk convention -- the rate of option-value
    DECAY as calendar time passes, i.e. theta = -d(price)/d(T), PER
    YEAR (divide by 365 for per-day theta). This is the opposite sign
    from the raw calculus derivative d(price)/d(T); see bs_theta's
    docstring for why this convention is used.

This module has no dependencies on the rest of the project -- every
function here is a pure function of (S, K, T, r, sigma, option_type).
"""

import math
from dataclasses import dataclass


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _validate_S_K(S, K):
    """Shared input guard: the underlying price and strike must both be
    strictly positive (S/K appears inside a log, and both represent
    real, positive-valued prices)."""
    if S <= 0:
        raise ValueError(f"S (underlying price) must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"K (strike price) must be positive, got {K}")


def _validate_option_type(option_type):
    """Shared input guard, checked before any Black-Scholes computation
    so an invalid option_type fails fast rather than after wasted work."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _d1_d2(S, K, T, r, sigma):
    """Black-Scholes d1 and d2 terms. Only ever called from branches
    that have already confirmed T > 0."""
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes price of a European call or put.

    At/after expiry (T <= 0) the option is settled at exact intrinsic
    value -- max(S-K, 0) for calls, max(K-S, 0) for puts -- rather than
    evaluated through the Black-Scholes formula with a near-zero time
    value. This keeps pricing well-behaved and exact right at maturity,
    with no dependence on sigma once T <= 0.
    """
    _validate_S_K(S, K)
    _validate_option_type(option_type)

    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """Option delta. At expiry, degenerates to 0/1 (call) or 0/-1 (put)
    based on moneyness, matching the intrinsic-value settlement above.

    Convention at the exact kink (S == K at expiry): delta is
    mathematically undefined there (intrinsic value has a corner), so
    by convention this returns 0.0 for both calls and puts rather than
    picking either one-sided limit."""
    _validate_S_K(S, K)
    _validate_option_type(option_type)

    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0  # S == K -> 0.0, by convention
        return -1.0 if S < K else 0.0     # S == K -> 0.0, by convention

    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    """Gamma is identical for calls and puts (put-call symmetry of d1).
    Zero at/after expiry: an intrinsic-value payoff has no curvature."""
    _validate_S_K(S, K)

    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    """Vega is identical for calls and puts. Zero at/after expiry: an
    expired option's intrinsic value no longer depends on volatility."""
    _validate_S_K(S, K)

    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)


def bs_theta(S, K, T, r, sigma, option_type="call"):
    """Theta, using the standard trading-desk convention: the rate of
    option-value DECAY as calendar time passes. Concretely,

        theta = -d(price)/d(T)

    NOT +d(price)/d(T). A no-dividend European option's price is
    monotonically increasing in time-to-maturity T (more time = more
    optionality = more value), so the raw calculus derivative
    d(price)/d(T) is positive. But "theta" as quoted on a trading desk
    (and as used elsewhere in this project, e.g. daily P&L attribution)
    means how much value is lost per unit of calendar time that passes
    -- which is the negative of that derivative. That's why theta is
    typically reported as a negative number for a long option, and why
    the formula below is `-` the naive d(price)/d(T) expansion.

    Expressed per year (divide by 365 for a per-calendar-day figure).

    Defined as exactly 0.0 at/after expiry (T <= 0) by implementation
    convention: once the option is settled at intrinsic value, there is
    no remaining time value left to decay, so there is nothing for
    theta to measure. This is a modeling convention for this project,
    not a general mathematical identity -- it simply reflects that
    intrinsic value has no time component.
    """
    _validate_S_K(S, K)
    _validate_option_type(option_type)

    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    common = -S * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    if option_type == "call":
        return common - r * K * math.exp(-r * T) * _norm_cdf(d2)
    return common + r * K * math.exp(-r * T) * _norm_cdf(-d2)


def bs_greeks(S, K, T, r, sigma, option_type="call") -> Greeks:
    """Convenience bundle of all four Greeks in one Black-Scholes call."""
    return Greeks(
        delta=bs_delta(S, K, T, r, sigma, option_type),
        gamma=bs_gamma(S, K, T, r, sigma),
        vega=bs_vega(S, K, T, r, sigma),
        theta=bs_theta(S, K, T, r, sigma, option_type),
    )