"""
tests/test_portfolio.py
========================
Unit tests for src/portfolio.py: Portfolio, PnLTracker,
check_pnl_conservation.

Run with:
    pytest tests/test_portfolio.py
or directly:
    python tests/test_portfolio.py

Scope: portfolio.py depends only on market.py (OptionContract,
OptionMarket) for valuation/Greeks -- these tests use that dependency
directly (a real OptionMarket, real Greeks) rather than mocking it,
since portfolio.py's whole job is correctly aggregating over it. They
do NOT retest pricing.py's Black-Scholes formulas themselves (covered
in test_pricing.py) or market_maker.py/hedging.py (portfolio.py has no
dependency on either -- confirmed structurally in section 8 below).

Sections:
    1. apply_option_fill (cash, inventory, cash-flow accumulator)
    2. apply_hedge_trade (cash, shares, commission, cash-flow accumulator)
    3. option_market_value
    4. net_option_delta / net_option_gamma / net_option_vega
    5. total_value
    6. PnLTracker.record (decomposition, accumulator reset, history)
    7. check_pnl_conservation (pass case, violation case, parameters)
    8. Structural: no dependency on market_maker.py / hedging.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd  # noqa: E402

from src.portfolio import Portfolio, PnLTracker, check_pnl_conservation  # noqa: E402
from src.market import OptionContract, OptionMarket  # noqa: E402


def _contract(strike=100, option_type="call"):
    return OptionContract(strike=strike, maturity=0.5, option_type=option_type)


def _market(pricing_vol=0.25, contracts=None):
    return OptionMarket(contracts or [_contract()], r=0.02, pricing_vol=pricing_vol)


# ============================================================
# 1. apply_option_fill
# ============================================================

def test_buy_increases_option_inventory():
    p = Portfolio()
    c = _contract()
    p.apply_option_fill(c, side="buy", price=10.0, qty=5)
    assert p.option_qty[c] == 5


def test_sell_decreases_option_inventory():
    p = Portfolio()
    c = _contract()
    p.apply_option_fill(c, side="sell", price=10.0, qty=5)
    assert p.option_qty[c] == -5


def test_buy_decreases_cash():
    p = Portfolio()
    c = _contract()
    p.apply_option_fill(c, side="buy", price=10.0, qty=5)
    assert p.cash == -50.0


def test_sell_increases_cash():
    p = Portfolio()
    c = _contract()
    p.apply_option_fill(c, side="sell", price=10.0, qty=5)
    assert p.cash == 50.0


def test_multiple_fills_on_same_contract_net_correctly():
    p = Portfolio()
    c = _contract()
    p.apply_option_fill(c, side="buy", price=10.0, qty=5)
    p.apply_option_fill(c, side="sell", price=11.0, qty=2)
    assert p.option_qty[c] == 3
    assert p.cash == -50.0 + 22.0


def test_fills_on_different_contracts_tracked_separately():
    p = Portfolio()
    c1, c2 = _contract(strike=100), _contract(strike=110)
    p.apply_option_fill(c1, side="buy", price=10.0, qty=5)
    p.apply_option_fill(c2, side="sell", price=3.0, qty=2)
    assert p.option_qty[c1] == 5
    assert p.option_qty[c2] == -2


def test_apply_option_fill_rejects_invalid_side():
    p = Portfolio()
    c = _contract()
    try:
        p.apply_option_fill(c, side="hold", price=10.0, qty=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_option_cash_flow_accumulator_tracks_fills():
    p = Portfolio()
    c = _contract()
    assert p.option_cash_flow == 0.0
    p.apply_option_fill(c, side="sell", price=10.0, qty=5)
    assert p.option_cash_flow == 50.0
    p.apply_option_fill(c, side="buy", price=9.0, qty=2)
    assert p.option_cash_flow == 50.0 - 18.0


def test_option_fill_does_not_affect_hedge_cash_flow():
    p = Portfolio()
    c = _contract()
    p.apply_option_fill(c, side="sell", price=10.0, qty=5)
    assert p.hedge_cash_flow == 0.0


# ============================================================
# 2. apply_hedge_trade
# ============================================================

def test_positive_qty_increases_shares():
    p = Portfolio()
    p.apply_hedge_trade(qty=10.0, price=100.0, commission=0.0)
    assert p.shares == 10.0


def test_negative_qty_decreases_shares():
    p = Portfolio()
    p.apply_hedge_trade(qty=-10.0, price=100.0, commission=0.0)
    assert p.shares == -10.0


def test_buying_shares_decreases_cash_by_price_times_qty_plus_commission():
    p = Portfolio()
    p.apply_hedge_trade(qty=10.0, price=100.0, commission=0.5)
    assert p.cash == -(10.0 * 100.0) - 0.5


def test_selling_shares_increases_cash_minus_commission():
    p = Portfolio()
    p.apply_hedge_trade(qty=-10.0, price=100.0, commission=0.5)
    # flow = -qty*price - commission = -(-10*100) - 0.5 = 1000 - 0.5
    assert p.cash == 1000.0 - 0.5


def test_commission_defaults_to_zero():
    p = Portfolio()
    p.apply_hedge_trade(qty=5.0, price=100.0)
    assert p.cash == -500.0


def test_hedge_cash_flow_accumulator_tracks_trades():
    p = Portfolio()
    assert p.hedge_cash_flow == 0.0
    p.apply_hedge_trade(qty=10.0, price=100.0, commission=0.5)
    assert p.hedge_cash_flow == -1000.5


def test_hedge_trade_does_not_affect_option_cash_flow():
    p = Portfolio()
    p.apply_hedge_trade(qty=10.0, price=100.0, commission=0.5)
    assert p.option_cash_flow == 0.0


def test_zero_qty_hedge_trade_still_charges_commission_if_given():
    """apply_hedge_trade doesn't special-case qty=0 -- if a caller
    passes qty=0 with a nonzero commission, that commission is still
    charged. (In practice, hedging.py returns commission=0.0 whenever
    it returns qty=0.0, so this combination shouldn't arise from normal
    use -- this test just documents Portfolio's own behavior in
    isolation, without assuming anything about hedging.py's caller
    discipline.)"""
    p = Portfolio()
    p.apply_hedge_trade(qty=0.0, price=100.0, commission=1.0)
    assert p.shares == 0.0
    assert p.cash == -1.0


# ============================================================
# 3. option_market_value
# ============================================================

def test_option_market_value_zero_with_no_positions():
    p = Portfolio()
    market = _market()
    assert p.option_market_value(market, S=100.0, t=0.0) == 0.0


def test_option_market_value_matches_manual_sum():
    p = Portfolio()
    c1, c2 = _contract(strike=95), _contract(strike=105)
    market = _market(contracts=[c1, c2])
    p.apply_option_fill(c1, side="buy", price=1.0, qty=4)
    p.apply_option_fill(c2, side="sell", price=1.0, qty=3)
    S, t = 100.0, 0.1
    expected = 4 * market.theo_value(c1, S, t) + (-3) * market.theo_value(c2, S, t)
    assert p.option_market_value(market, S, t) == expected


def test_option_market_value_ignores_flattened_positions():
    """A contract bought then fully sold back nets to qty=0; it's still
    a dict entry, but should contribute exactly 0 to market value."""
    p = Portfolio()
    c = _contract()
    market = _market()
    p.apply_option_fill(c, side="buy", price=10.0, qty=5)
    p.apply_option_fill(c, side="sell", price=11.0, qty=5)
    assert p.option_qty[c] == 0
    assert p.option_market_value(market, S=100.0, t=0.0) == 0.0


# ============================================================
# 4. net_option_delta / net_option_gamma / net_option_vega
# ============================================================

def test_net_option_delta_matches_manual_sum():
    p = Portfolio()
    c1, c2 = _contract(strike=95), _contract(strike=105)
    market = _market(contracts=[c1, c2])
    p.apply_option_fill(c1, side="buy", price=1.0, qty=4)
    p.apply_option_fill(c2, side="sell", price=1.0, qty=3)
    S, t = 100.0, 0.1
    expected = 4 * market.greeks(c1, S, t).delta + (-3) * market.greeks(c2, S, t).delta
    assert p.net_option_delta(market, S, t) == expected


def test_net_option_delta_zero_with_no_positions():
    p = Portfolio()
    market = _market()
    assert p.net_option_delta(market, S=100.0, t=0.0) == 0.0


def test_net_option_gamma_matches_manual_sum():
    p = Portfolio()
    c = _contract()
    market = _market()
    p.apply_option_fill(c, side="buy", price=5.0, qty=6)
    S, t = 100.0, 0.1
    expected = 6 * market.greeks(c, S, t).gamma
    assert p.net_option_gamma(market, S, t) == expected


def test_net_option_vega_matches_manual_sum():
    p = Portfolio()
    c = _contract()
    market = _market()
    p.apply_option_fill(c, side="sell", price=5.0, qty=6)
    S, t = 100.0, 0.1
    expected = -6 * market.greeks(c, S, t).vega
    assert p.net_option_vega(market, S, t) == expected


def test_long_call_has_positive_delta_gamma_vega():
    """Sanity check tying the aggregation back to known option
    properties: a long call position should have positive delta,
    gamma, and vega (all of which are individually positive for a
    single call, per pricing.py, so a long position preserves sign)."""
    p = Portfolio()
    c = _contract()
    market = _market()
    p.apply_option_fill(c, side="buy", price=5.0, qty=1)
    S, t = 100.0, 0.1
    assert p.net_option_delta(market, S, t) > 0
    assert p.net_option_gamma(market, S, t) > 0
    assert p.net_option_vega(market, S, t) > 0


# ============================================================
# 5. total_value
# ============================================================

def test_total_value_of_empty_portfolio_is_zero():
    p = Portfolio()
    market = _market()
    assert p.total_value(market, S=100.0, t=0.0) == 0.0


def test_total_value_combines_cash_options_and_shares():
    p = Portfolio()
    c = _contract()
    market = _market()
    p.apply_option_fill(c, side="buy", price=5.0, qty=2)   # cash -10
    p.apply_hedge_trade(qty=3.0, price=100.0, commission=1.0)  # cash -301
    S, t = 100.0, 0.1
    expected = p.cash + p.option_market_value(market, S, t) + p.shares * S
    assert p.total_value(market, S, t) == expected
    assert p.cash == -10.0 - 301.0
    assert p.shares == 3.0


# ============================================================
# 6. PnLTracker.record
# ============================================================

def test_record_returns_option_and_shares_market_value():
    p = Portfolio()
    c = _contract()
    market = _market()
    p.apply_option_fill(c, side="buy", price=5.0, qty=2)
    tracker = PnLTracker()
    option_mv, shares_mv = tracker.record(0.0, 100.0, p, market, 0.0, 0.0)
    assert option_mv == p.option_market_value(market, 100.0, 0.0)
    assert shares_mv == 0.0  # no shares held yet


def test_record_resets_cash_flow_accumulators():
    p = Portfolio()
    c = _contract()
    market = _market()
    p.apply_option_fill(c, side="sell", price=10.0, qty=5)
    p.apply_hedge_trade(qty=2.0, price=100.0, commission=0.1)
    assert p.option_cash_flow != 0.0
    assert p.hedge_cash_flow != 0.0
    tracker = PnLTracker()
    tracker.record(0.0, 100.0, p, market, 0.0, 0.0)
    assert p.option_cash_flow == 0.0
    assert p.hedge_cash_flow == 0.0


def test_record_appends_one_entry_per_call():
    p = Portfolio()
    market = _market()
    tracker = PnLTracker()
    tracker.record(0.0, 100.0, p, market, 0.0, 0.0)
    tracker.record(1 / 252, 101.0, p, market, 0.0, 0.0)
    assert len(tracker.history) == 2


def test_total_pnl_step_equals_sum_of_option_and_hedge_pnl():
    p = Portfolio()
    c = _contract()
    market = _market()
    p.apply_option_fill(c, side="sell", price=10.0, qty=5)
    tracker = PnLTracker()
    tracker.record(0.0, 100.0, p, market, 0.0, 0.0)
    entry = tracker.history[0]
    assert entry["total_pnl_step"] == entry["option_pnl_step"] + entry["hedge_pnl_step"]


def test_option_only_fill_produces_zero_hedge_pnl_that_step():
    """A step with only an option fill and no shares held should
    produce hedge_pnl_step == 0 -- nothing happened on the hedge side."""
    p = Portfolio()
    c = _contract()
    market = _market()
    p.apply_option_fill(c, side="sell", price=10.0, qty=5)
    tracker = PnLTracker()
    tracker.record(0.0, 100.0, p, market, 0.0, 0.0)
    assert tracker.history[0]["hedge_pnl_step"] == 0.0


def test_no_activity_step_produces_zero_pnl_if_price_unchanged():
    """If nothing traded and the underlying didn't move, and there are
    no open positions, P&L for that step should be exactly zero."""
    p = Portfolio()
    market = _market()
    tracker = PnLTracker()
    tracker.record(0.0, 100.0, p, market, 0.0, 0.0)
    assert tracker.history[0]["total_pnl_step"] == 0.0


# ============================================================
# 7. check_pnl_conservation
# ============================================================

def test_conservation_passes_for_a_correctly_ordered_multi_step_scenario():
    p = Portfolio()
    c = _contract()
    market = _market()
    tracker = PnLTracker()
    prev_mv, prev_smv = 0.0, 0.0
    total_values = []

    p.apply_option_fill(c, side="sell", price=10.0, qty=5)
    prev_mv, prev_smv = tracker.record(0.0, 100.0, p, market, prev_mv, prev_smv)
    total_values.append(p.total_value(market, 100.0, 0.0))

    delta_exp = p.net_option_delta(market, 103.0, 1 / 252)
    p.apply_hedge_trade(qty=-delta_exp, price=103.0, commission=0.05)
    prev_mv, prev_smv = tracker.record(1 / 252, 103.0, p, market, prev_mv, prev_smv)
    total_values.append(p.total_value(market, 103.0, 1 / 252))

    p.apply_option_fill(c, side="buy", price=12.0, qty=2)
    prev_mv, prev_smv = tracker.record(2 / 252, 104.0, p, market, prev_mv, prev_smv)
    total_values.append(p.total_value(market, 104.0, 2 / 252))

    pnl_df = pd.DataFrame(tracker.history)
    max_diff = check_pnl_conservation(pd.Series(total_values), pnl_df["total_pnl_step"])
    assert max_diff < 1e-9


def test_conservation_raises_when_timing_convention_is_violated():
    """record() called BEFORE a fill that total_value_series then
    includes -- violates the documented ordering and must be caught."""
    p = Portfolio()
    c = _contract()
    market = _market()
    tracker = PnLTracker()
    prev_mv, prev_smv = 0.0, 0.0
    total_values = []

    p.apply_option_fill(c, side="sell", price=10.0, qty=5)
    prev_mv, prev_smv = tracker.record(0.0, 100.0, p, market, prev_mv, prev_smv)
    total_values.append(p.total_value(market, 100.0, 0.0))

    # record() first, fill applied afterward -- misordered
    prev_mv, prev_smv = tracker.record(1 / 252, 103.0, p, market, prev_mv, prev_smv)
    p.apply_option_fill(c, side="buy", price=9.0, qty=1)
    total_values.append(p.total_value(market, 103.0, 1 / 252))

    pnl_df = pd.DataFrame(tracker.history)
    try:
        check_pnl_conservation(pd.Series(total_values), pnl_df["total_pnl_step"])
        assert False, "expected AssertionError for misordered fill/record"
    except AssertionError:
        pass


def test_conservation_uses_initial_value_parameter_not_first_entry():
    """A scenario deliberately starting from a nonzero initial_value
    (e.g. a portfolio pre-seeded with cash) should reconcile correctly
    when initial_value is passed explicitly, and fail the default
    (0.0) check if it's omitted."""
    p = Portfolio()
    p.cash = 500.0  # pre-seed cash directly, bypassing fills
    market = _market()
    tracker = PnLTracker()
    prev_mv, prev_smv = 0.0, 0.0
    total_values = []

    p.apply_hedge_trade(qty=2.0, price=100.0, commission=0.0)
    prev_mv, prev_smv = tracker.record(0.0, 100.0, p, market, prev_mv, prev_smv)
    total_values.append(p.total_value(market, 100.0, 0.0))

    pnl_df = pd.DataFrame(tracker.history)

    # With the correct initial_value (500.0), conservation holds.
    max_diff = check_pnl_conservation(
        pd.Series(total_values), pnl_df["total_pnl_step"], initial_value=500.0)
    assert max_diff < 1e-9

    # With the (wrong, default) initial_value=0.0, it should NOT hold,
    # since total_value already includes the pre-seeded 500 cash that
    # was never captured as a "pnl step".
    try:
        check_pnl_conservation(pd.Series(total_values), pnl_df["total_pnl_step"])
        assert False, "expected AssertionError with mismatched initial_value"
    except AssertionError:
        pass


def test_conservation_respects_tolerance_parameter():
    """A tiny, deliberately-introduced discrepancy should pass with a
    loose tolerance and fail with a tight one."""
    total_values = pd.Series([100.0, 100.00005])
    pnl_steps = pd.Series([100.0, 0.0])  # off by 0.00005 on step 2
    assert check_pnl_conservation(total_values, pnl_steps, tol=1e-3) < 1e-3
    try:
        check_pnl_conservation(total_values, pnl_steps, tol=1e-6)
        assert False, "expected AssertionError with tight tolerance"
    except AssertionError:
        pass


def test_conservation_returns_the_max_absolute_discrepancy():
    total_values = pd.Series([100.0, 100.0])
    pnl_steps = pd.Series([100.0, 0.0])
    max_diff = check_pnl_conservation(total_values, pnl_steps, tol=1.0)
    assert abs(max_diff - 0.0) < 1e-9


# ============================================================
# 8. Structural: no dependency on market_maker.py / hedging.py
# ============================================================

def test_portfolio_module_does_not_import_market_maker_or_hedging():
    """portfolio.py's update methods take plain parameters (contract,
    side, price, qty, commission), never Fill/HedgeTrade objects.
    Confirmed here by parsing the module's actual import statements
    (via ast) -- not a raw text search, since the module's own
    docstring legitimately MENTIONS market_maker.py/hedging.py by name
    when explaining why there's no dependency on them."""
    import ast
    import src.portfolio as portfolio_module

    with open(portfolio_module.__file__) as f:
        tree = ast.parse(f.read())

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    assert not any("market_maker" in m for m in imported_modules)
    assert not any("hedging" in m for m in imported_modules)


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