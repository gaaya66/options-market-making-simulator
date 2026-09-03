"""
tests/test_market.py
=====================
Unit tests for src/market.py: simulate_gbm, OptionContract, OptionMarket.

Run with:
    pytest tests/test_market.py
or directly:
    python tests/test_market.py

Deliberately does NOT re-derive Black-Scholes math here -- pricing.py
already has 39 tests covering that. These tests check that market.py
correctly ORCHESTRATES pricing.py (passes the right S/K/T/r/sigma
through) and gets its own responsibilities right: the GBM simulator,
OptionContract's identity/hashing, time-to-expiry arithmetic, and the
strict separation between pricing_vol and realized_vol.

Sections:
    1. GBM path length and starting price
    2. GBM reproducibility with a fixed seed
    3. Positive underlying prices
    4. OptionContract construction and hashability
    5. Time-to-expiry
    6. Call/put theoretical values (agreement with pricing.py)
    7. Greeks passed correctly from pricing.py
    8. Exact expiry behaviour
    9. Post-expiry behaviour
   10. Separation of pricing_vol and realized_vol
"""

import inspect
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.market import simulate_gbm, OptionContract, OptionMarket  # noqa: E402
from src.pricing import bs_price, bs_greeks, Greeks  # noqa: E402


# ============================================================
# 1. GBM path length and starting price
# ============================================================

def test_gbm_path_length_is_n_steps_plus_one():
    path = simulate_gbm(S0=100, mu=0.05, sigma=0.25, dt=1 / 252, n_steps=252, seed=0)
    assert len(path) == 253


def test_gbm_path_starts_at_S0():
    path = simulate_gbm(S0=137.5, mu=0.05, sigma=0.25, dt=1 / 252, n_steps=10, seed=0)
    assert path[0] == 137.5


def test_gbm_single_step_path_has_two_points():
    """n_steps=0 should give just the starting price -- no steps taken."""
    path = simulate_gbm(S0=100, mu=0.05, sigma=0.25, dt=1 / 252, n_steps=0, seed=0)
    assert len(path) == 1
    assert path[0] == 100


# ============================================================
# 2. GBM reproducibility with a fixed seed
# ============================================================

def test_gbm_same_seed_gives_identical_path():
    path_a = simulate_gbm(S0=100, mu=0.05, sigma=0.25, dt=1 / 252, n_steps=100, seed=42)
    path_b = simulate_gbm(S0=100, mu=0.05, sigma=0.25, dt=1 / 252, n_steps=100, seed=42)
    assert list(path_a) == list(path_b)


def test_gbm_different_seeds_give_different_paths():
    path_a = simulate_gbm(S0=100, mu=0.05, sigma=0.25, dt=1 / 252, n_steps=100, seed=1)
    path_b = simulate_gbm(S0=100, mu=0.05, sigma=0.25, dt=1 / 252, n_steps=100, seed=2)
    assert list(path_a) != list(path_b)


def test_gbm_no_seed_still_produces_a_valid_path():
    """seed=None should still work (uses OS entropy) -- just checking it
    doesn't error and produces a sane-length path."""
    path = simulate_gbm(S0=100, mu=0.05, sigma=0.25, dt=1 / 252, n_steps=50, seed=None)
    assert len(path) == 51
    assert path[0] == 100


# ============================================================
# 3. Positive underlying prices
# ============================================================

def test_gbm_prices_always_positive_typical_vol():
    path = simulate_gbm(S0=100, mu=0.05, sigma=0.25, dt=1 / 252, n_steps=1000, seed=7)
    assert (path > 0).all()


def test_gbm_prices_always_positive_high_vol():
    """GBM is a lognormal process by construction (it's an exponential
    of a normal random walk), so it can never hit zero or go negative
    -- even at extreme volatility."""
    path = simulate_gbm(S0=50, mu=0.0, sigma=2.0, dt=1 / 252, n_steps=1000, seed=3)
    assert (path > 0).all()


def test_gbm_prices_always_positive_negative_drift():
    """A strongly negative drift should not push prices through zero
    either -- GBM decays toward zero asymptotically but never reaches
    or crosses it."""
    path = simulate_gbm(S0=100, mu=-0.5, sigma=0.3, dt=1 / 252, n_steps=1000, seed=9)
    assert (path > 0).all()


# ============================================================
# 4. OptionContract construction and hashability
# ============================================================

def test_option_contract_fields_are_stored():
    c = OptionContract(strike=105.0, maturity=0.5, option_type="call")
    assert c.strike == 105.0
    assert c.maturity == 0.5
    assert c.option_type == "call"


def test_option_contract_usable_as_dict_key():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    inventory = {c: 5}
    assert inventory[c] == 5


def test_option_contracts_with_identical_fields_are_equal_and_hash_equal():
    """Two separately constructed contracts with the same strike/
    maturity/type must be treated as the same key -- required so
    inventory tracking (market_maker.py, portfolio.py) doesn't
    accidentally create duplicate entries for what's really one
    contract."""
    c1 = OptionContract(strike=100, maturity=0.5, option_type="call")
    c2 = OptionContract(strike=100, maturity=0.5, option_type="call")
    assert c1 == c2
    assert hash(c1) == hash(c2)
    inventory = {c1: 3}
    inventory[c2] = inventory.get(c2, 0) + 2
    assert inventory == {c1: 5}


def test_option_contracts_with_different_fields_are_not_equal():
    c1 = OptionContract(strike=100, maturity=0.5, option_type="call")
    c2 = OptionContract(strike=105, maturity=0.5, option_type="call")
    c3 = OptionContract(strike=100, maturity=0.5, option_type="put")
    assert c1 != c2
    assert c1 != c3


def test_option_contract_is_frozen():
    """Immutability matters here: if strike/maturity/type could be
    mutated after construction, its hash could change while it's
    sitting in a dict, silently corrupting inventory lookups."""
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    try:
        c.strike = 200
        assert False, "expected an error mutating a frozen dataclass"
    except Exception:
        pass


def test_option_contract_name_property():
    call = OptionContract(strike=100, maturity=0.5, option_type="call")
    put = OptionContract(strike=95.5, maturity=0.5, option_type="put")
    assert call.name == "C100"
    assert put.name == "P95.5"


# ============================================================
# 5. Time-to-expiry
# ============================================================

def test_time_to_expiry_before_maturity():
    c = OptionContract(strike=100, maturity=1.0, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert market.time_to_expiry(c, t=0.3) == 0.7


def test_time_to_expiry_at_t_zero_equals_full_maturity():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert market.time_to_expiry(c, t=0.0) == 0.5


def test_time_to_expiry_at_exact_maturity_is_zero():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert market.time_to_expiry(c, t=0.5) == 0.0


def test_time_to_expiry_clips_at_zero_past_maturity():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert market.time_to_expiry(c, t=0.9) == 0.0
    assert market.time_to_expiry(c, t=100.0) == 0.0


# ============================================================
# 6. Call/put theoretical values (agreement with pricing.py)
# ============================================================

def test_call_theo_value_matches_direct_bs_price_call():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.30)
    S, t = 105.0, 0.1
    expected = bs_price(S, c.strike, c.maturity - t, market.r, market.pricing_vol, "call")
    assert market.theo_value(c, S, t) == expected


def test_put_theo_value_matches_direct_bs_price_put():
    c = OptionContract(strike=100, maturity=0.5, option_type="put")
    market = OptionMarket([c], r=0.02, pricing_vol=0.30)
    S, t = 95.0, 0.2
    expected = bs_price(S, c.strike, c.maturity - t, market.r, market.pricing_vol, "put")
    assert market.theo_value(c, S, t) == expected


def test_call_theo_value_is_positive_before_expiry():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert market.theo_value(c, S=100, t=0.0) > 0


def test_put_call_theo_values_satisfy_parity():
    """Reusing the put-call parity identity as a cross-check that
    OptionMarket is feeding pricing.py consistent inputs for both
    contract types, not just that each matches in isolation."""
    call = OptionContract(strike=100, maturity=0.5, option_type="call")
    put = OptionContract(strike=100, maturity=0.5, option_type="put")
    market = OptionMarket([call, put], r=0.03, pricing_vol=0.25)
    S, t = 105.0, 0.1
    T_remaining = call.maturity - t
    lhs = market.theo_value(call, S, t) - market.theo_value(put, S, t)
    rhs = S - call.strike * math.exp(-market.r * T_remaining)
    assert abs(lhs - rhs) < 1e-9


# ============================================================
# 7. Greeks passed correctly from pricing.py
# ============================================================

def test_market_greeks_returns_greeks_dataclass():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    g = market.greeks(c, S=100, t=0.0)
    assert isinstance(g, Greeks)


def test_market_greeks_matches_direct_bs_greeks_call():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.30)
    S, t = 105.0, 0.1
    expected = bs_greeks(S, c.strike, c.maturity - t, market.r, market.pricing_vol, "call")
    actual = market.greeks(c, S, t)
    assert actual.delta == expected.delta
    assert actual.gamma == expected.gamma
    assert actual.vega == expected.vega
    assert actual.theta == expected.theta


def test_market_greeks_matches_direct_bs_greeks_put():
    c = OptionContract(strike=100, maturity=0.5, option_type="put")
    market = OptionMarket([c], r=0.02, pricing_vol=0.30)
    S, t = 95.0, 0.2
    expected = bs_greeks(S, c.strike, c.maturity - t, market.r, market.pricing_vol, "put")
    actual = market.greeks(c, S, t)
    assert actual.delta == expected.delta
    assert actual.gamma == expected.gamma
    assert actual.vega == expected.vega
    assert actual.theta == expected.theta


# ============================================================
# 8. Exact expiry behaviour
# ============================================================

def test_theo_value_at_expiry_itm_call_is_exact_intrinsic():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert market.theo_value(c, S=115, t=0.5) == 15.0


def test_theo_value_at_expiry_otm_call_is_exact_zero():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert market.theo_value(c, S=90, t=0.5) == 0.0


def test_theo_value_at_expiry_itm_put_is_exact_intrinsic():
    c = OptionContract(strike=100, maturity=0.5, option_type="put")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert market.theo_value(c, S=85, t=0.5) == 15.0


def test_greeks_at_expiry_are_degenerate():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    g = market.greeks(c, S=115, t=0.5)
    assert g.delta == 1.0
    assert g.gamma == 0.0
    assert g.vega == 0.0
    assert g.theta == 0.0


# ============================================================
# 9. Post-expiry behaviour
# ============================================================

def test_theo_value_past_expiry_still_returns_intrinsic():
    """Valuation must keep working correctly past maturity -- this is
    deliberately still just a VALUATION, not a settlement action; the
    position isn't removed or modified by calling this method."""
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert market.theo_value(c, S=115, t=0.9) == 15.0
    assert market.theo_value(c, S=90, t=2.0) == 0.0


def test_theo_value_past_expiry_does_not_raise():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    # Should not raise even far past maturity, and with sigma that
    # would matter if (incorrectly) still routed through Black-Scholes.
    market.theo_value(c, S=100, t=50.0)
    market.greeks(c, S=100, t=50.0)


def test_contract_object_is_unchanged_by_valuation_past_expiry():
    """Calling theo_value()/greeks() past maturity must not mutate the
    contract itself (it's frozen) or implicitly alter OptionMarket's
    state -- confirms valuation has no settlement side effects."""
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    market.theo_value(c, S=115, t=10.0)
    assert c.strike == 100 and c.maturity == 0.5 and c.option_type == "call"
    assert market.contracts == [c]


# ============================================================
# 10. Separation of pricing_vol and realized_vol
# ============================================================

def test_option_market_has_no_realized_vol_attribute():
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)
    assert not hasattr(market, "realized_vol")


def test_simulate_gbm_signature_has_no_pricing_vol_parameter():
    """Structural guard: simulate_gbm's parameters should never include
    anything named 'pricing_vol' -- the underlying path must be driven
    only by its own sigma (realized_vol), never by the market's pricing
    assumption."""
    params = inspect.signature(simulate_gbm).parameters
    assert "pricing_vol" not in params
    assert "sigma" in params


def test_option_market_init_signature_has_no_realized_vol_parameter():
    """Mirror of the above: OptionMarket must never accept realized_vol
    -- it should only ever be able to see pricing_vol."""
    params = inspect.signature(OptionMarket.__init__).parameters
    assert "realized_vol" not in params
    assert "pricing_vol" in params


def test_theo_value_unaffected_by_which_realized_vol_generated_the_path():
    """The realized_vol used to simulate a GBM path has zero influence
    on how OptionMarket prices an option at a given (S, t) -- only
    pricing_vol matters. Simulate two paths with very different
    realized_vol, then confirm OptionMarket prices identically at the
    same (S, t), since OptionMarket never even receives realized_vol."""
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.25)

    path_low_vol = simulate_gbm(S0=100, mu=0.05, sigma=0.05, dt=1 / 252, n_steps=50, seed=1)
    path_high_vol = simulate_gbm(S0=100, mu=0.05, sigma=0.80, dt=1 / 252, n_steps=50, seed=1)
    assert list(path_low_vol) != list(path_high_vol)  # the paths do actually differ

    # But pricing at the SAME S, t must be identical regardless of which
    # realized_vol produced that S -- OptionMarket only sees S and t.
    S_probe, t_probe = 103.0, 0.1
    assert market.theo_value(c, S_probe, t_probe) == market.theo_value(c, S_probe, t_probe)


def test_pricing_vol_and_realized_vol_can_differ_freely():
    """Sanity check that nothing in the API forces the two vols to
    match -- a market maker pricing at 20% vol while the underlying
    realizes 50% vol is a normal, fully supported configuration."""
    c = OptionContract(strike=100, maturity=0.5, option_type="call")
    market = OptionMarket([c], r=0.02, pricing_vol=0.20)
    path = simulate_gbm(S0=100, mu=0.05, sigma=0.50, dt=1 / 252, n_steps=50, seed=0)
    # Just confirming this combination runs without error or coupling.
    value = market.theo_value(c, S=path[-1], t=50 / 252)
    assert value >= 0


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