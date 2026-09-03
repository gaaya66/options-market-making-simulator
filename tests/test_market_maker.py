"""
tests/test_market_maker.py
===========================
Unit tests for src/market_maker.py: MarketMaker, OrderFlowGenerator,
Quote, Fill.

Run with:
    pytest tests/test_market_maker.py
or directly:
    python tests/test_market_maker.py

Scope: these tests are about market_maker.py's own responsibilities --
quoting arithmetic, inventory skew, inventory-limit enforcement (both
at quote time and at execution time), the customer-order/MM-side
convention, and the order-flow generator. They deliberately do NOT
retest anything already covered in test_pricing.py or test_market.py
(no Black-Scholes math, no OptionMarket/theo_value/time-to-expiry
behaviour) -- OptionContract is used here only as a convenient,
already-tested hashable key, not as a subject under test.

Sections:
    1. Bid/ask calculation
    2. Inventory skew (long and short)
    3. Zero-inventory quotes
    4. Bid/ask non-negativity and ordering
    5. Maximum inventory limits (quote-time)
    6. Execution: buy increases inventory, sell decreases it
    7. Execution: inventory-cap enforcement (fill-time clipping)
    8. Customer buy/sell <-> MM side convention
    9. Poisson order-flow generation
   10. Reproducibility with a fixed seed
   11. Quote and Fill data structures
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.market_maker import MarketMaker, OrderFlowGenerator, Quote, Fill  # noqa: E402
from src.market import OptionContract  # noqa: E402


def _contract():
    """A single reusable contract -- market_maker.py doesn't care what
    it prices, only that it's a stable, hashable inventory key."""
    return OptionContract(strike=100, maturity=0.5, option_type="call")


# ============================================================
# 1. Bid/ask calculation
# ============================================================

def test_bid_ask_at_zero_inventory_matches_half_spread_formula():
    """With no inventory (no skew), bid/ask should be exactly
    theo -/+ theo*half_spread."""
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.002, max_inventory=10)
    q = mm.quote(_contract(), theo=100.0)
    assert q.bid == 95.0
    assert q.ask == 105.0


def test_bid_ask_scales_with_theo_value():
    mm = MarketMaker(half_spread=0.10, inventory_skew=0.0, max_inventory=10)
    q = mm.quote(_contract(), theo=50.0)
    assert q.bid == 45.0
    assert q.ask == 55.0


def test_quote_sizes_equal_configured_quote_size_when_uncapped():
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=10, quote_size=3)
    q = mm.quote(_contract(), theo=100.0)
    assert q.bid_size == 3
    assert q.ask_size == 3


# ============================================================
# 2. Inventory skew (long and short)
# ============================================================

def test_long_inventory_shifts_both_quotes_down():
    """A positive (long) inventory should push both bid and ask DOWN by
    the same amount, relative to the zero-inventory quote -- making the
    MM a worse buyer and a cheaper (more attractive) seller."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.002, max_inventory=50)
    q_flat = mm.quote(c, theo=100.0)  # inventory is 0 here

    mm.inventory[c] = 10  # simulate being long 10
    q_long = mm.quote(c, theo=100.0)

    expected_skew = 100.0 * 0.002 * 10  # = 2.0
    assert q_long.bid == q_flat.bid - expected_skew
    assert q_long.ask == q_flat.ask - expected_skew


def test_short_inventory_shifts_both_quotes_up():
    """The mirror image: a negative (short) inventory should push both
    quotes UP -- making the MM a more attractive buyer (to cover the
    short) and a worse seller."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.002, max_inventory=50)
    q_flat = mm.quote(c, theo=100.0)

    mm.inventory[c] = -10  # simulate being short 10
    q_short = mm.quote(c, theo=100.0)

    expected_skew = 100.0 * 0.002 * (-10)  # = -2.0
    assert q_short.bid == q_flat.bid - expected_skew
    assert q_short.ask == q_flat.ask - expected_skew


def test_larger_inventory_produces_larger_skew():
    """Skew is linear in inventory: doubling the position should
    exactly double the shift away from the flat-inventory quote."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.002, max_inventory=50)
    q_flat = mm.quote(c, theo=100.0)

    mm.inventory[c] = 5
    shift_5 = q_flat.bid - mm.quote(c, theo=100.0).bid

    mm.inventory[c] = 10
    shift_10 = q_flat.bid - mm.quote(c, theo=100.0).bid

    assert abs(shift_10 - 2 * shift_5) < 1e-9


# ============================================================
# 3. Zero-inventory quotes
# ============================================================

def test_zero_inventory_quote_has_no_skew_applied():
    """Explicitly confirms a freshly-created MarketMaker (no trades
    yet) quotes with zero skew, i.e. get_inventory defaults to 0."""
    c = _contract()
    mm = MarketMaker(half_spread=0.04, inventory_skew=0.01, max_inventory=10)
    assert mm.get_inventory(c) == 0
    q = mm.quote(c, theo=200.0)
    assert q.bid == 200.0 * 0.96
    assert q.ask == 200.0 * 1.04


# ============================================================
# 4. Bid/ask non-negativity and ordering
# ============================================================

def test_bid_never_negative_even_under_extreme_skew():
    """A very long position combined with a large skew coefficient
    could drive the raw bid formula deeply negative; quote() must
    floor it at 0.0 (a price can never be negative)."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=1.0, max_inventory=1000)
    mm.inventory[c] = 100  # deliberately extreme
    q = mm.quote(c, theo=1.0)
    assert q.bid >= 0.0


def test_ask_never_below_bid_even_under_extreme_skew():
    """Under the same extreme scenario, ask must never cross below bid
    -- quote() should floor ask at max(ask_raw, bid)."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=1.0, max_inventory=1000)
    mm.inventory[c] = 100
    q = mm.quote(c, theo=1.0)
    assert q.ask >= q.bid


def test_ask_at_or_above_bid_across_a_range_of_inventories():
    """General non-crossing property, checked across a spread of
    ordinary (non-extreme) inventory levels."""
    c = _contract()
    mm = MarketMaker(half_spread=0.03, inventory_skew=0.001, max_inventory=100)
    for inv in (-80, -20, -1, 0, 1, 20, 80):
        mm.inventory[c] = inv
        q = mm.quote(c, theo=100.0)
        assert q.ask >= q.bid


# ============================================================
# 5. Maximum inventory limits (quote-time)
# ============================================================

def test_bid_size_zero_when_at_positive_cap():
    """Once inventory == +max_inventory, quoting more bid size would
    let the position grow past the cap, so bid_size must be 0."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=5)
    mm.inventory[c] = 5
    q = mm.quote(c, theo=100.0)
    assert q.bid_size == 0
    assert q.ask_size > 0  # can still sell, to reduce the long


def test_ask_size_zero_when_at_negative_cap():
    """Mirror image: at -max_inventory, ask_size must be 0 (can't
    short further), but bid_size should still be available (to buy
    back and reduce the short)."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=5)
    mm.inventory[c] = -5
    q = mm.quote(c, theo=100.0)
    assert q.ask_size == 0
    assert q.bid_size > 0


def test_both_sides_available_when_inventory_within_bounds():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=5)
    mm.inventory[c] = 2
    q = mm.quote(c, theo=100.0)
    assert q.bid_size > 0
    assert q.ask_size > 0


# ============================================================
# 6. Execution: buy increases inventory, sell decreases it
# ============================================================

def test_execute_buy_increases_inventory():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=10)
    mm.execute(t=0.0, contract=c, side="buy", price=10.0, qty=3)
    assert mm.get_inventory(c) == 3


def test_execute_sell_decreases_inventory():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=10)
    mm.execute(t=0.0, contract=c, side="sell", price=10.0, qty=3)
    assert mm.get_inventory(c) == -3


def test_execute_buy_then_sell_nets_correctly():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=10)
    mm.execute(t=0.0, contract=c, side="buy", price=10.0, qty=5)
    mm.execute(t=0.0, contract=c, side="sell", price=10.5, qty=2)
    assert mm.get_inventory(c) == 3


def test_execute_rejects_invalid_side():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=10)
    try:
        mm.execute(t=0.0, contract=c, side="hold", price=10.0, qty=1)
        assert False, "expected ValueError for invalid side"
    except ValueError:
        pass


# ============================================================
# 7. Execution: inventory-cap enforcement (fill-time clipping)
# ============================================================

def test_execution_clips_to_exact_remaining_capacity():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=5, quote_size=100)
    fill = mm.execute(t=0.0, contract=c, side="sell", price=9.5, qty=5)
    assert fill.qty == 5
    assert mm.get_inventory(c) == -5


def test_execution_clips_partial_fill_when_only_one_remains():
    """The exact scenario from the design discussion: 1 unit of
    capacity remains before the cap, an order for 5 arrives -- only 1
    should execute, not 5."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=5, quote_size=100)
    mm.execute(t=0.0, contract=c, side="sell", price=9.5, qty=4)  # inv -> -4
    fill = mm.execute(t=0.0, contract=c, side="sell", price=9.5, qty=5)  # only 1 room left
    assert fill.qty == 1
    assert mm.get_inventory(c) == -5


def test_execution_fills_zero_once_cap_is_reached():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=5, quote_size=100)
    mm.execute(t=0.0, contract=c, side="sell", price=9.5, qty=5)  # at cap
    fill = mm.execute(t=0.0, contract=c, side="sell", price=9.5, qty=10)
    assert fill.qty == 0
    assert mm.get_inventory(c) == -5


def test_execution_clipping_applies_symmetrically_to_buy_side():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=4, quote_size=100)
    fill = mm.execute(t=0.0, contract=c, side="buy", price=10.5, qty=100)
    assert fill.qty == 4
    assert mm.get_inventory(c) == 4


def test_inventory_never_exceeds_cap_across_many_random_sized_orders():
    """A stress-style check: repeatedly hit the same side with
    oversized orders and confirm inventory is monotonically clamped at
    the cap, never overshooting it even transiently."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=7, quote_size=100)
    for _ in range(20):
        mm.execute(t=0.0, contract=c, side="sell", price=9.5, qty=3)
        assert -7 <= mm.get_inventory(c) <= 7
    assert mm.get_inventory(c) == -7


# ============================================================
# 8. Customer buy/sell <-> MM side convention
# ============================================================

def test_customer_buy_maps_to_mm_side_sell():
    flow = OrderFlowGenerator(intensity=3.0, buy_prob=1.0, seed=0)  # force all customer buys
    orders = flow.generate([_contract()])
    assert len(orders) > 0
    assert all(side == "sell" for _, side, _ in orders)


def test_customer_sell_maps_to_mm_side_buy():
    flow = OrderFlowGenerator(intensity=3.0, buy_prob=0.0, seed=0)  # force all customer sells
    orders = flow.generate([_contract()])
    assert len(orders) > 0
    assert all(side == "buy" for _, side, _ in orders)


def test_end_to_end_customer_buy_fills_at_ask_and_reduces_mm_position():
    """Integration-style check tying the convention to an actual quote
    and fill: a customer buy should fill against the MM's ask, and the
    MM's inventory should decrease (the MM sold)."""
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=10)
    flow = OrderFlowGenerator(intensity=5.0, buy_prob=1.0, seed=1)
    orders = flow.generate([c])
    contract, mm_side, qty = orders[0]
    q = mm.quote(contract, theo=100.0)
    assert mm_side == "sell"
    fill = mm.execute(t=0.0, contract=contract, side=mm_side, price=q.ask, qty=qty)
    assert fill.price == q.ask
    assert mm.get_inventory(c) == -qty


def test_end_to_end_customer_sell_fills_at_bid_and_increases_mm_position():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=10)
    flow = OrderFlowGenerator(intensity=5.0, buy_prob=0.0, seed=1)
    orders = flow.generate([c])
    contract, mm_side, qty = orders[0]
    q = mm.quote(contract, theo=100.0)
    assert mm_side == "buy"
    fill = mm.execute(t=0.0, contract=contract, side=mm_side, price=q.bid, qty=qty)
    assert fill.price == q.bid
    assert mm.get_inventory(c) == qty


# ============================================================
# 9. Poisson order-flow generation
# ============================================================

def test_zero_intensity_generates_no_orders():
    flow = OrderFlowGenerator(intensity=0.0, buy_prob=0.5, seed=0)
    orders = flow.generate([_contract(), _contract()])
    assert orders == []


def test_generated_orders_use_configured_size():
    flow = OrderFlowGenerator(intensity=5.0, buy_prob=0.5, size=7, seed=0)
    orders = flow.generate([_contract()])
    assert len(orders) > 0
    assert all(qty == 7 for _, _, qty in orders)


def test_generated_orders_reference_the_given_contracts():
    c1 = OptionContract(strike=100, maturity=0.5, option_type="call")
    c2 = OptionContract(strike=110, maturity=0.5, option_type="put")
    flow = OrderFlowGenerator(intensity=3.0, buy_prob=0.5, seed=0)
    orders = flow.generate([c1, c2])
    for contract, _, _ in orders:
        assert contract in (c1, c2)


def test_average_order_count_is_close_to_intensity():
    """Statistical check: over many independent timesteps, the average
    number of orders per contract per step should converge toward the
    configured Poisson intensity (law of large numbers), within a
    generous tolerance to keep the test non-flaky."""
    flow = OrderFlowGenerator(intensity=2.0, buy_prob=0.5, seed=0)
    c = _contract()
    counts = [len(flow.generate([c])) for _ in range(5000)]
    avg = sum(counts) / len(counts)
    assert abs(avg - 2.0) < 0.15


# ============================================================
# 10. Reproducibility with a fixed seed
# ============================================================

def test_same_seed_produces_identical_order_sequence():
    c = _contract()
    flow_a = OrderFlowGenerator(intensity=2.0, buy_prob=0.5, seed=123)
    flow_b = OrderFlowGenerator(intensity=2.0, buy_prob=0.5, seed=123)
    orders_a = [flow_a.generate([c]) for _ in range(20)]
    orders_b = [flow_b.generate([c]) for _ in range(20)]
    assert orders_a == orders_b


def test_different_seeds_produce_different_order_sequences():
    c = _contract()
    flow_a = OrderFlowGenerator(intensity=2.0, buy_prob=0.5, seed=1)
    flow_b = OrderFlowGenerator(intensity=2.0, buy_prob=0.5, seed=2)
    orders_a = [flow_a.generate([c]) for _ in range(20)]
    orders_b = [flow_b.generate([c]) for _ in range(20)]
    assert orders_a != orders_b


# ============================================================
# 11. Quote and Fill data structures
# ============================================================

def test_quote_fields_are_accessible():
    q = Quote(bid=95.0, ask=105.0, bid_size=1, ask_size=1)
    assert q.bid == 95.0
    assert q.ask == 105.0
    assert q.bid_size == 1
    assert q.ask_size == 1


def test_quote_equality_by_value():
    """Quote is a plain dataclass -- two instances with identical field
    values should compare equal (useful for assertions elsewhere)."""
    q1 = Quote(bid=95.0, ask=105.0, bid_size=1, ask_size=1)
    q2 = Quote(bid=95.0, ask=105.0, bid_size=1, ask_size=1)
    assert q1 == q2


def test_fill_fields_are_accessible():
    c = _contract()
    f = Fill(t=1.5, contract=c, side="buy", price=10.0, qty=3)
    assert f.t == 1.5
    assert f.contract is c
    assert f.side == "buy"
    assert f.price == 10.0
    assert f.qty == 3


def test_execute_returns_a_fill_instance():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=10)
    fill = mm.execute(t=0.0, contract=c, side="buy", price=10.0, qty=1)
    assert isinstance(fill, Fill)


def test_quote_returns_a_quote_instance():
    c = _contract()
    mm = MarketMaker(half_spread=0.05, inventory_skew=0.0, max_inventory=10)
    q = mm.quote(c, theo=100.0)
    assert isinstance(q, Quote)


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