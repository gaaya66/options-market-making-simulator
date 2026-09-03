"""
tests/test_hedging.py
======================
Unit tests for src/hedging.py: DeltaHedger, HedgeTrade.

Run with:
    pytest tests/test_hedging.py
or directly:
    python tests/test_hedging.py

Scope: hedging.py is deliberately decoupled from pricing.py, market.py,
and market_maker.py -- it only ever receives plain floats
(option_delta_exposure, S, shares_held). These tests exercise it in
exactly that decoupled way, with made-up delta numbers, and never
import OptionContract/OptionMarket/MarketMaker at all.

Sections:
    1. should_hedge cadence
    2. Already-hedged case (no trade needed)
    3. Direction of the hedge trade (buy vs. sell)
    4. Trade size (full re-hedge to zero)
    5. Spread cost embedded in execution price
    6. Commission accounting
    7. HedgeTrade data structure
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hedging import DeltaHedger, HedgeTrade  # noqa: E402


# ============================================================
# 1. should_hedge cadence
# ============================================================

def test_should_hedge_every_step_when_hedge_every_is_one():
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    assert all(hedger.should_hedge(i) for i in range(10))


def test_should_hedge_fires_on_multiples_of_hedge_every():
    hedger = DeltaHedger(hedge_every=5, spread_bps=5.0)
    fires = [i for i in range(21) if hedger.should_hedge(i)]
    assert fires == [0, 5, 10, 15, 20]


def test_should_hedge_includes_step_zero():
    """The book should start hedged from the very first timestep, not
    drift unhedged until the first interval completes."""
    hedger = DeltaHedger(hedge_every=21, spread_bps=5.0)
    assert hedger.should_hedge(0) is True


def test_should_hedge_false_between_intervals():
    hedger = DeltaHedger(hedge_every=10, spread_bps=5.0)
    assert hedger.should_hedge(3) is False
    assert hedger.should_hedge(9) is False
    assert hedger.should_hedge(11) is False


# ============================================================
# 2. Already-hedged case
# ============================================================

def test_no_trade_when_already_hedged():
    """If option delta exposure is exactly 0 and no shares are held,
    net delta is already 0 -- hedge() should propose zero trade."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0, commission_per_share=0.01)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=0.0, shares_held=0.0)
    assert trade.qty == 0.0
    assert trade.commission == 0.0


def test_no_trade_when_shares_already_offset_delta():
    """Net delta can be 0 even with nonzero option delta, if shares
    already offset it exactly -- hedge() should recognize this and not
    trade again."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0, commission_per_share=0.01)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=15.0, shares_held=-15.0)
    assert trade.qty == 0.0
    assert trade.commission == 0.0


def test_no_trade_price_defaults_to_underlying_mid():
    """When no trade happens, price should just be the current
    underlying price S (no spread applied, since nothing executed)."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    trade = hedger.hedge(t=0.0, S=137.25, option_delta_exposure=0.0, shares_held=0.0)
    assert trade.price == 137.25


# ============================================================
# 3. Direction of the hedge trade
# ============================================================

def test_positive_option_delta_triggers_a_sell():
    """A long option delta exposure means the hedger must sell shares
    to bring net delta back to zero."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=12.0, shares_held=0.0)
    assert trade.qty < 0


def test_negative_option_delta_triggers_a_buy():
    """A short (negative) option delta exposure means the hedger must
    buy shares to bring net delta back to zero."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=-12.0, shares_held=0.0)
    assert trade.qty > 0


# ============================================================
# 4. Trade size (full re-hedge to zero)
# ============================================================

def test_trade_size_fully_offsets_option_delta_from_flat():
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=23.0, shares_held=0.0)
    assert trade.qty == -23.0


def test_trade_size_accounts_for_shares_already_held():
    """If some shares are already held, the trade should only cover
    the REMAINING gap to full delta-neutrality, not the whole exposure
    again."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    # option delta = 20, already holding -12 shares -> need -8 more
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=20.0, shares_held=-12.0)
    assert trade.qty == -8.0


def test_trade_size_can_overshoot_correction_if_overhedged():
    """If shares_held already over-corrects net delta, the hedger
    should trade back the other way to return to exactly zero, not
    just stop trading."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    # option delta = 5, but holding -20 shares (way over-hedged short)
    # -> current net delta = 5 + (-20) = -15, so the hedger must BUY 15
    # shares to bring net delta back to exactly 0.
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=5.0, shares_held=-20.0)
    assert trade.qty == 15.0


def test_result_is_always_exactly_delta_neutral():
    """General property: after applying the returned qty to
    shares_held, option_delta_exposure + (shares_held + qty) should
    equal exactly 0, for a range of scenarios."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    scenarios = [(10.0, 0.0), (-7.5, 3.0), (0.3, -0.3), (100.0, -40.0)]
    for delta_exp, shares in scenarios:
        trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=delta_exp, shares_held=shares)
        assert abs(delta_exp + (shares + trade.qty)) < 1e-9


# ============================================================
# 5. Spread cost embedded in execution price
# ============================================================

def test_buy_executes_above_mid_price():
    """Buying should cross the ask side: price = S + half_spread."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=10.0)  # 10 bps
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=-5.0, shares_held=0.0)
    expected_price = 100.0 + 100.0 * 10.0 / 10_000.0
    assert trade.price == expected_price


def test_sell_executes_below_mid_price():
    """Selling should cross the bid side: price = S - half_spread."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=10.0)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=5.0, shares_held=0.0)
    expected_price = 100.0 - 100.0 * 10.0 / 10_000.0
    assert trade.price == expected_price


def test_zero_spread_executes_exactly_at_mid():
    hedger = DeltaHedger(hedge_every=1, spread_bps=0.0)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=5.0, shares_held=0.0)
    assert trade.price == 100.0


def test_spread_scales_with_underlying_price():
    """Spread is in basis points of S, so a higher S should widen the
    absolute spread proportionally."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=20.0)
    trade_low = hedger.hedge(t=0.0, S=50.0, option_delta_exposure=5.0, shares_held=0.0)
    trade_high = hedger.hedge(t=0.0, S=500.0, option_delta_exposure=5.0, shares_held=0.0)
    spread_low = 50.0 - trade_low.price
    spread_high = 500.0 - trade_high.price
    assert abs(spread_high - 10 * spread_low) < 1e-9


# ============================================================
# 6. Commission accounting
# ============================================================

def test_commission_scales_with_trade_size():
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0, commission_per_share=0.02)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=25.0, shares_held=0.0)
    assert abs(trade.commission - (25.0 * 0.02)) < 1e-9


def test_commission_uses_absolute_trade_size_for_sells_too():
    """Commission should be based on |qty|, not signed qty -- a sell of
    25 shares costs the same commission as a buy of 25 shares."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0, commission_per_share=0.02)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=25.0, shares_held=0.0)  # sell
    assert trade.commission > 0
    assert abs(trade.commission - (abs(trade.qty) * 0.02)) < 1e-9


def test_zero_commission_rate_gives_zero_commission():
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0, commission_per_share=0.0)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=25.0, shares_held=0.0)
    assert trade.commission == 0.0


def test_default_commission_rate_is_zero():
    """commission_per_share should default to 0.0 if not specified."""
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=25.0, shares_held=0.0)
    assert trade.commission == 0.0


# ============================================================
# 7. HedgeTrade data structure
# ============================================================

def test_hedge_returns_a_hedgetrade_instance():
    hedger = DeltaHedger(hedge_every=1, spread_bps=5.0)
    trade = hedger.hedge(t=0.0, S=100.0, option_delta_exposure=5.0, shares_held=0.0)
    assert isinstance(trade, HedgeTrade)


def test_hedgetrade_fields_are_accessible():
    trade = HedgeTrade(t=1.5, qty=-3.0, price=99.9, commission=0.06)
    assert trade.t == 1.5
    assert trade.qty == -3.0
    assert trade.price == 99.9
    assert trade.commission == 0.06


def test_hedgetrade_uses_commission_field_not_cost():
    """Explicit check that the field is named `commission`, per the
    accounting-clarity change from the design discussion -- not `cost`
    (which could be confused with total spread + commission cost)."""
    trade = HedgeTrade(t=0.0, qty=1.0, price=100.0, commission=0.05)
    assert hasattr(trade, "commission")
    assert not hasattr(trade, "cost")


def test_hedgetrade_equality_by_value():
    t1 = HedgeTrade(t=0.0, qty=5.0, price=100.0, commission=0.1)
    t2 = HedgeTrade(t=0.0, qty=5.0, price=100.0, commission=0.1)
    assert t1 == t2


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