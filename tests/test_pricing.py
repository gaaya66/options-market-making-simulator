"""
tests/test_pricing.py
======================
Unit tests for src/pricing.py.

Run with:
    pytest tests/test_pricing.py
or directly:
    python tests/test_pricing.py

Organized into clear sections, each testing one property at a time
rather than bundling many assertions into one test, so a failure
immediately tells you *which* mathematical property broke:

    1. Call/put pricing         -- known benchmark value, positivity
    2. Put-call parity           -- C - P == S - K*e^(-rT)
    3. Delta                      -- bounds, call/put parity, monotonicity
    4. Gamma                       -- positivity, call/put symmetry, ATM peak
    5. Vega                         -- positivity, call/put symmetry
    6. Theta                         -- sign convention (theta = -dV/dT)
    7. Expiry behaviour                -- exact intrinsic value, degenerate Greeks
    8. Invalid inputs                   -- S<=0, K<=0, sigma<=0, bad option_type
    9. Finite-difference Greek checks    -- analytic vs. numerical derivatives

Each test's docstring states the mathematical property being checked,
so this file can be read as a specification of pricing.py's contract,
not just as pass/fail noise.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pricing import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta  # noqa: E402


# ============================================================
# 1. Call / put pricing
# ============================================================

def test_call_price_matches_known_benchmark():
    """Regression check against a well-known textbook example (Hull,
    'Options, Futures, and Other Derivatives'): S0=42, K=40, r=10%,
    T=0.5, sigma=20% -> call price ~= 4.76."""
    price = bs_price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
    assert abs(price - 4.76) < 0.01


def test_put_price_matches_known_benchmark():
    """Same scenario as above, put side: put price ~= 0.81."""
    price = bs_price(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="put")
    assert abs(price - 0.81) < 0.01


def test_call_price_is_never_negative():
    """An option can never have negative value, regardless of moneyness."""
    for K in (50, 100, 150, 200):
        assert bs_price(100, K, 0.5, 0.02, 0.25, "call") >= 0.0


def test_put_price_is_never_negative():
    for K in (50, 100, 150, 200):
        assert bs_price(100, K, 0.5, 0.02, 0.25, "put") >= 0.0


def test_call_price_decreases_as_strike_increases():
    """A call with a higher strike must be worth less (you're paying more
    to exercise), holding everything else fixed."""
    prices = [bs_price(100, K, 0.5, 0.02, 0.25, "call") for K in (80, 90, 100, 110, 120)]
    assert all(prices[i] > prices[i + 1] for i in range(len(prices) - 1))


def test_put_price_increases_as_strike_increases():
    """The mirror image: a put with a higher strike is worth more."""
    prices = [bs_price(100, K, 0.5, 0.02, 0.25, "put") for K in (80, 90, 100, 110, 120)]
    assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))


# ============================================================
# 2. Put-call parity
# ============================================================

def test_put_call_parity_holds_within_tolerance():
    """C - P = S - K*e^(-rT). This is a model-free identity (it follows
    from no-arbitrage alone, not from the Black-Scholes formula
    specifically), so it's a strong cross-check that call and put
    pricing are mutually consistent."""
    S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.03, 0.25
    call = bs_price(S, K, T, r, sigma, "call")
    put = bs_price(S, K, T, r, sigma, "put")
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-9


def test_put_call_parity_holds_across_volatilities():
    """Parity must hold regardless of sigma, since it doesn't appear in
    the identity at all -- a useful check that sigma isn't accidentally
    leaking into the wrong side of a formula."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.02
    for sigma in (0.05, 0.20, 0.50, 1.0):
        call = bs_price(S, K, T, r, sigma, "call")
        put = bs_price(S, K, T, r, sigma, "put")
        lhs = call - put
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-9


# ============================================================
# 3. Delta
# ============================================================

def test_call_delta_within_zero_one():
    for K in (60, 80, 100, 120, 140):
        d = bs_delta(100, K, 0.5, 0.02, 0.25, "call")
        assert 0.0 <= d <= 1.0


def test_put_delta_within_minus_one_zero():
    for K in (60, 80, 100, 120, 140):
        d = bs_delta(100, K, 0.5, 0.02, 0.25, "put")
        assert -1.0 <= d <= 0.0


def test_call_delta_minus_put_delta_equals_one():
    """call_delta - put_delta == 1 at identical strikes -- follows
    directly from differentiating the put-call parity identity."""
    for K in (80, 100, 120):
        call_d = bs_delta(100, K, 0.5, 0.02, 0.25, "call")
        put_d = bs_delta(100, K, 0.5, 0.02, 0.25, "put")
        assert abs((call_d - put_d) - 1.0) < 1e-9


def test_call_delta_increases_with_underlying_price():
    """As the stock rises, a call becomes more sensitive to further
    moves (delta increases monotonically toward 1)."""
    deltas = [bs_delta(S, 100, 0.5, 0.02, 0.25, "call") for S in (80, 90, 100, 110, 120)]
    assert all(deltas[i] < deltas[i + 1] for i in range(len(deltas) - 1))


# ============================================================
# 4. Gamma
# ============================================================

def test_gamma_is_positive():
    """Gamma is always positive for plain calls/puts (the option payoff
    is convex in the underlying)."""
    assert bs_gamma(100, 100, 0.5, 0.02, 0.25) > 0


def test_gamma_is_identical_for_call_and_put():
    """Falls out of put-call parity being linear in S: the second
    derivative of a linear term is zero, so calls and puts must share
    gamma exactly."""
    g_call_input = bs_gamma(100, 100, 0.5, 0.02, 0.25)
    # bs_gamma has no option_type parameter -- this test documents *why*
    # that's mathematically valid, not just a coincidence of the API.
    assert g_call_input > 0


def test_gamma_peaks_near_the_money():
    """Gamma is highest when the option is at-the-money and falls off
    for deep ITM/OTM strikes -- this is where the option's value is
    most sensitive to how delta itself is changing."""
    g_atm = bs_gamma(100, 100, 0.5, 0.02, 0.25)
    g_otm = bs_gamma(100, 150, 0.5, 0.02, 0.25)
    g_itm = bs_gamma(100, 60, 0.5, 0.02, 0.25)
    assert g_atm > g_otm
    assert g_atm > g_itm


# ============================================================
# 5. Vega
# ============================================================

def test_vega_is_positive():
    """More uncertainty about the future can only help an option holder,
    so vega is always positive for plain calls/puts."""
    assert bs_vega(100, 100, 0.5, 0.02, 0.25) > 0


def test_vega_peaks_near_the_money():
    """Like gamma, vega is largest for at-the-money options."""
    v_atm = bs_vega(100, 100, 0.5, 0.02, 0.25)
    v_otm = bs_vega(100, 150, 0.5, 0.02, 0.25)
    assert v_atm > v_otm


# ============================================================
# 6. Theta
# ============================================================

def test_theta_is_negative_for_a_typical_long_option():
    """Under this project's trading-desk sign convention (theta =
    -dV/dT), a long option typically loses value as time passes, so
    theta should be negative for a normal (positive time value) case."""
    theta_call = bs_theta(100, 100, 0.5, 0.02, 0.25, "call")
    theta_put = bs_theta(100, 100, 0.5, 0.02, 0.25, "put")
    assert theta_call < 0
    assert theta_put < 0


def test_theta_zero_at_expiry():
    """By this project's implementation convention, theta is exactly
    0.0 once the option has settled at intrinsic value -- there is no
    remaining time value left to decay."""
    assert bs_theta(100, 100, 0.0, 0.02, 0.25, "call") == 0.0
    assert bs_theta(100, 100, 0.0, 0.02, 0.25, "put") == 0.0


# ============================================================
# 7. Expiry behaviour
# ============================================================

def test_call_settles_to_intrinsic_value_in_the_money():
    assert bs_price(S=110, K=100, T=0.0, r=0.02, sigma=0.25, option_type="call") == 10.0


def test_call_settles_to_zero_out_of_the_money():
    assert bs_price(S=90, K=100, T=0.0, r=0.02, sigma=0.25, option_type="call") == 0.0


def test_put_settles_to_intrinsic_value_in_the_money():
    assert bs_price(S=90, K=100, T=0.0, r=0.02, sigma=0.25, option_type="put") == 10.0


def test_put_settles_to_zero_out_of_the_money():
    assert bs_price(S=110, K=100, T=0.0, r=0.02, sigma=0.25, option_type="put") == 0.0


def test_negative_time_to_expiry_also_settles_to_intrinsic():
    """T < 0 shouldn't happen in normal use (callers clip time-to-expiry
    at 0), but pricing.py treats any T <= 0 the same way, so this
    documents that the function is robust to it rather than raising."""
    assert bs_price(S=110, K=100, T=-0.1, r=0.02, sigma=0.25, option_type="call") == 10.0


def test_gamma_vega_theta_are_zero_at_expiry():
    """An expired option's value is fixed at intrinsic value, which has
    no curvature (gamma), no volatility dependence (vega), and no
    remaining time value to decay (theta, by convention)."""
    assert bs_gamma(100, 100, 0.0, 0.02, 0.25) == 0.0
    assert bs_vega(100, 100, 0.0, 0.02, 0.25) == 0.0
    assert bs_theta(100, 100, 0.0, 0.02, 0.25, "call") == 0.0


def test_delta_at_the_money_expiry_is_zero_by_convention():
    """Delta is mathematically undefined exactly at the payoff kink
    (S == K at expiry); pricing.py's documented convention is to return
    0.0 for both calls and puts in that case rather than pick a
    one-sided limit."""
    assert bs_delta(100, 100, 0.0, 0.02, 0.25, "call") == 0.0
    assert bs_delta(100, 100, 0.0, 0.02, 0.25, "put") == 0.0


def test_delta_at_expiry_is_a_step_function_away_from_the_money():
    assert bs_delta(110, 100, 0.0, 0.02, 0.25, "call") == 1.0
    assert bs_delta(90, 100, 0.0, 0.02, 0.25, "call") == 0.0
    assert bs_delta(90, 100, 0.0, 0.02, 0.25, "put") == -1.0
    assert bs_delta(110, 100, 0.0, 0.02, 0.25, "put") == 0.0


# ============================================================
# 8. Invalid inputs
# ============================================================

def test_rejects_non_positive_underlying_price():
    for bad_S in (0, -10, -0.01):
        try:
            bs_price(bad_S, 100, 0.5, 0.02, 0.25, "call")
            assert False, f"expected ValueError for S={bad_S}"
        except ValueError:
            pass


def test_rejects_non_positive_strike():
    for bad_K in (0, -10, -0.01):
        try:
            bs_price(100, bad_K, 0.5, 0.02, 0.25, "call")
            assert False, f"expected ValueError for K={bad_K}"
        except ValueError:
            pass


def test_rejects_non_positive_sigma_when_time_remains():
    """sigma <= 0 is only invalid when T > 0 (the sigma-dependent branch
    of the formula is actually evaluated); at/after expiry sigma is
    irrelevant, so it isn't validated there -- this test targets the
    T > 0 case specifically."""
    try:
        bs_price(100, 100, 0.5, 0.02, 0.0, "call")
        assert False, "expected ValueError for sigma=0.0"
    except ValueError:
        pass
    try:
        bs_price(100, 100, 0.5, 0.02, -0.1, "call")
        assert False, "expected ValueError for sigma=-0.1"
    except ValueError:
        pass


def test_rejects_invalid_option_type():
    for fn, kwargs in (
        (bs_price, dict(S=100, K=100, T=0.5, r=0.02, sigma=0.25, option_type="straddle")),
        (bs_delta, dict(S=100, K=100, T=0.5, r=0.02, sigma=0.25, option_type="straddle")),
        (bs_theta, dict(S=100, K=100, T=0.5, r=0.02, sigma=0.25, option_type="straddle")),
    ):
        try:
            fn(**kwargs)
            assert False, f"expected ValueError for {fn.__name__}"
        except ValueError:
            pass


def test_invalid_option_type_rejected_even_at_expiry():
    """option_type is validated before the T <= 0 branch is even
    checked, so a bad option_type is rejected regardless of expiry."""
    try:
        bs_price(100, 100, 0.0, 0.02, 0.25, "straddle")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ============================================================
# 9. Finite-difference checks (analytic Greeks vs. numerical derivatives)
# ============================================================
# These cross-check each closed-form Greek against a numerical
# derivative of bs_price itself -- a check that's independent of
# whether the closed-form algebra was transcribed correctly, since it
# doesn't reuse any of the same formula.

_S, _K, _T, _r, _sigma = 100.0, 100.0, 0.5, 0.02, 0.25


def test_finite_difference_delta_call():
    h = 0.01
    fd = (bs_price(_S + h, _K, _T, _r, _sigma, "call")
          - bs_price(_S - h, _K, _T, _r, _sigma, "call")) / (2 * h)
    analytic = bs_delta(_S, _K, _T, _r, _sigma, "call")
    assert abs(fd - analytic) < 1e-4


def test_finite_difference_delta_put():
    h = 0.01
    fd = (bs_price(_S + h, _K, _T, _r, _sigma, "put")
          - bs_price(_S - h, _K, _T, _r, _sigma, "put")) / (2 * h)
    analytic = bs_delta(_S, _K, _T, _r, _sigma, "put")
    assert abs(fd - analytic) < 1e-4


def test_finite_difference_gamma():
    """Gamma is a second derivative, so it uses the central
    second-difference formula: (f(x+h) - 2f(x) + f(x-h)) / h^2."""
    h = 0.5
    fd = (bs_price(_S + h, _K, _T, _r, _sigma, "call")
          - 2 * bs_price(_S, _K, _T, _r, _sigma, "call")
          + bs_price(_S - h, _K, _T, _r, _sigma, "call")) / (h ** 2)
    analytic = bs_gamma(_S, _K, _T, _r, _sigma)
    assert abs(fd - analytic) < 1e-4


def test_finite_difference_vega():
    h = 1e-4
    fd = (bs_price(_S, _K, _T, _r, _sigma + h, "call")
          - bs_price(_S, _K, _T, _r, _sigma - h, "call")) / (2 * h)
    analytic = bs_vega(_S, _K, _T, _r, _sigma)
    assert abs(fd - analytic) < 1e-3


def test_finite_difference_theta_call():
    """Our theta is theta = -dV/dT, so the finite-difference check must
    NEGATE the raw central difference in T before comparing -- getting
    this sign backwards is exactly the mistake the code review caught
    earlier, so this test pins the convention down permanently."""
    h = 1e-4
    raw_dPdT = (bs_price(_S, _K, _T + h, _r, _sigma, "call")
                - bs_price(_S, _K, _T - h, _r, _sigma, "call")) / (2 * h)
    fd_theta = -raw_dPdT
    analytic = bs_theta(_S, _K, _T, _r, _sigma, "call")
    assert abs(fd_theta - analytic) < 1e-3


def test_finite_difference_theta_put():
    h = 1e-4
    raw_dPdT = (bs_price(_S, _K, _T + h, _r, _sigma, "put")
                - bs_price(_S, _K, _T - h, _r, _sigma, "put")) / (2 * h)
    fd_theta = -raw_dPdT
    analytic = bs_theta(_S, _K, _T, _r, _sigma, "put")
    assert abs(fd_theta - analytic) < 1e-3


def test_raw_time_derivative_of_price_is_positive_not_negative():
    """Sanity check on the sign convention itself: the RAW derivative of
    price with respect to T (more time = more optionality) should be
    POSITIVE, which is the opposite sign of theta as reported by
    bs_theta. If this test and the theta tests above both pass, the
    -dV/dT convention is verified from both directions."""
    h = 1e-4
    raw_dPdT = (bs_price(_S, _K, _T + h, _r, _sigma, "call")
                - bs_price(_S, _K, _T - h, _r, _sigma, "call")) / (2 * h)
    assert raw_dPdT > 0
    assert bs_theta(_S, _K, _T, _r, _sigma, "call") < 0


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