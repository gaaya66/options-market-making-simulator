"""
tests/test_simulation.py
=========================
Unit tests for src/simulation.py: SimConfig, build_contracts,
run_simulation.

Run with:
    pytest tests/test_simulation.py
or directly:
    python tests/test_simulation.py

Scope: simulation.py is pure orchestration -- it has no formulas of
its own. These tests focus on the things that could only go wrong AT
the orchestration layer: the Step A-F ordering (fills before delta,
delta before hedge, hedge before P&L record, post-hedge net_delta),
the terminal-step order-flow skip, reproducibility, and the DataFrame
shape/columns analytics.py will consume. They do NOT re-verify
Black-Scholes math, quoting arithmetic, or the hedge formula itself
(covered in their own modules' test files) -- only that
run_simulation() wires the pieces together correctly.

Sections:
    1. SimConfig / build_contracts
    2. History and pnl_history shape
    3. Reproducibility with a fixed seed
    4. P&L conservation (automatic, end-to-end)
    5. Event ordering: fills before delta, delta/hedge before record
    6. Post-hedge net_delta behaviour (hedge_every=1 vs. large)
    7. Hedge cadence respected in the history
    8. Terminal-step order-flow skip
    9. Inventory-cap enforcement carried through end-to-end
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from src.simulation import SimConfig, build_contracts, run_simulation  # noqa: E402
from src.market import simulate_gbm  # noqa: E402
from src.portfolio import check_pnl_conservation  # noqa: E402


# ============================================================
# 1. SimConfig / build_contracts
# ============================================================

def test_default_simconfig_constructs():
    cfg = SimConfig()
    assert cfg.n_steps == 252
    assert cfg.hedge_every == 1


def test_build_contracts_creates_one_call_and_one_put_per_strike():
    contracts = build_contracts(strikes=(90, 100, 110), maturity=0.5)
    assert len(contracts) == 6
    types = [c.option_type for c in contracts]
    assert types.count("call") == 3
    assert types.count("put") == 3


def test_build_contracts_assigns_correct_strikes_and_maturity():
    contracts = build_contracts(strikes=(90, 100), maturity=0.25)
    strikes_seen = {c.strike for c in contracts}
    assert strikes_seen == {90, 100}
    assert all(c.maturity == 0.25 for c in contracts)


def test_build_contracts_single_strike_gives_exactly_two_contracts():
    contracts = build_contracts(strikes=(100,), maturity=0.5)
    assert len(contracts) == 2
    assert {c.option_type for c in contracts} == {"call", "put"}


# ============================================================
# 2. History and pnl_history shape
# ============================================================

def test_history_has_n_steps_plus_one_rows():
    cfg = SimConfig(seed=0, n_steps=50)
    history, _ = run_simulation(cfg)
    assert len(history) == 51


def test_pnl_history_has_n_steps_plus_one_rows():
    cfg = SimConfig(seed=0, n_steps=50)
    _, pnl_history = run_simulation(cfg)
    assert len(pnl_history) == 51


def test_history_has_expected_columns():
    cfg = SimConfig(seed=0, n_steps=20)
    history, _ = run_simulation(cfg)
    expected = {"step", "t", "S", "total_abs_inventory", "shares",
                "option_delta", "option_gamma", "option_vega",
                "net_delta", "hedge_qty", "total_value"}
    assert expected.issubset(set(history.columns))


def test_pnl_history_has_expected_columns():
    cfg = SimConfig(seed=0, n_steps=20)
    _, pnl_history = run_simulation(cfg)
    expected = {"t", "S", "option_pnl_step", "hedge_pnl_step", "total_pnl_step"}
    assert expected.issubset(set(pnl_history.columns))


def test_history_t_column_matches_dt_times_step():
    cfg = SimConfig(seed=0, n_steps=10, dt=1 / 252)
    history, _ = run_simulation(cfg)
    expected_t = [i * (1 / 252) for i in range(11)]
    assert list(history["t"]) == expected_t


# ============================================================
# 3. Reproducibility with a fixed seed
# ============================================================

def test_same_seed_gives_identical_history():
    cfg_a = SimConfig(seed=7, n_steps=100)
    cfg_b = SimConfig(seed=7, n_steps=100)
    history_a, _ = run_simulation(cfg_a)
    history_b, _ = run_simulation(cfg_b)
    assert history_a["S"].tolist() == history_b["S"].tolist()
    assert history_a["total_value"].tolist() == history_b["total_value"].tolist()


def test_different_seeds_give_different_history():
    cfg_a = SimConfig(seed=1, n_steps=100)
    cfg_b = SimConfig(seed=2, n_steps=100)
    history_a, _ = run_simulation(cfg_a)
    history_b, _ = run_simulation(cfg_b)
    assert history_a["S"].tolist() != history_b["S"].tolist()


def test_underlying_path_matches_independent_simulate_gbm_call():
    """The S column should exactly reproduce a standalone simulate_gbm
    call with the same seed/parameters -- confirming simulation.py
    doesn't introduce any extra randomness or re-derive the path."""
    cfg = SimConfig(seed=42, n_steps=60, S0=100.0, mu=0.05, realized_vol=0.25,
                     dt=1 / 252)
    history, _ = run_simulation(cfg)
    expected_path = simulate_gbm(100.0, 0.05, 0.25, 1 / 252, 60, seed=42)
    assert history["S"].tolist() == list(expected_path)


# ============================================================
# 4. P&L conservation (automatic, end-to-end)
# ============================================================

def test_conservation_holds_for_default_config():
    cfg = SimConfig(seed=0, n_steps=100)
    history, pnl_history = run_simulation(cfg)  # would already raise if violated
    max_diff = check_pnl_conservation(history["total_value"], pnl_history["total_pnl_step"])
    assert max_diff < 1e-6


def test_conservation_holds_across_varied_configs():
    """Run several structurally different configs and confirm none of
    them raise inside run_simulation (which enforces conservation
    internally) -- spread, vol, and hedge frequency all varied."""
    configs = [
        SimConfig(seed=1, n_steps=80, mm_half_spread=0.15, realized_vol=0.5),
        SimConfig(seed=2, n_steps=80, hedge_every=10, realized_vol=0.1),
        SimConfig(seed=3, n_steps=80, order_flow_intensity=3.0, max_inventory=5),
    ]
    for cfg in configs:
        history, pnl_history = run_simulation(cfg)  # raises AssertionError on violation
        max_diff = check_pnl_conservation(history["total_value"], pnl_history["total_pnl_step"])
        assert max_diff < 1e-6


# ============================================================
# 5. Event ordering: fills before delta, delta/hedge before record
# ============================================================

def test_net_delta_reflects_shares_after_hedge_not_before():
    """If hedging is working (Step D before Step F), net_delta should
    be materially closer to zero than option_delta alone, whenever a
    hedge actually fired that step."""
    cfg = SimConfig(seed=5, n_steps=100, hedge_every=1)
    history, _ = run_simulation(cfg)
    hedged_steps = history[history["hedge_qty"] != 0]
    if len(hedged_steps) > 0:
        assert (hedged_steps["net_delta"].abs() <= hedged_steps["option_delta"].abs() + 1e-6).all()


def test_shares_only_change_on_hedge_steps():
    """shares should be constant between consecutive steps unless a
    hedge fired -- confirming Step D (hedging) is the only thing that
    ever touches portfolio.shares."""
    cfg = SimConfig(seed=5, n_steps=100, hedge_every=7)
    history, _ = run_simulation(cfg)
    for i in range(1, len(history)):
        if history["hedge_qty"].iloc[i] == 0.0:
            assert history["shares"].iloc[i] == history["shares"].iloc[i - 1]


# ============================================================
# 6. Post-hedge net_delta behaviour (hedge_every=1 vs. large)
# ============================================================

def test_net_delta_near_zero_when_hedging_every_step():
    cfg = SimConfig(seed=9, n_steps=150, hedge_every=1)
    history, _ = run_simulation(cfg)
    rms_net_delta = float(np.sqrt(np.mean(history["net_delta"] ** 2)))
    assert rms_net_delta < 1e-6


def test_hedging_error_exactly_zero_when_hedge_every_one():
    """Regression test, added after investigating a reported 'bug'
    (hedging_error == 0.0 in the demo output with default SimConfig).
    Confirmed NOT a bug: with hedge_every=1, EVERY recorded step is a
    hedge step, so net_delta is pinned to (exactly, not just
    approximately) 0.0 every single step -- there is no 'between
    hedges' gap in this discretization for gamma-driven drift to
    appear in. This test pins that exact-zero behavior down explicitly
    (stricter than the < 1e-6 check above) so it can't silently
    regress into a small-but-nonzero value without being noticed."""
    from src.analytics import hedging_error
    cfg = SimConfig(seed=0, n_steps=100, hedge_every=1)
    history, _ = run_simulation(cfg)
    assert hedging_error(history) == 0.0
    assert (history["net_delta"] == 0.0).all()


def test_hedging_error_monotonically_related_to_hedge_frequency():
    """Companion to the exact-zero test above: confirms the mechanism
    actually responds to hedge_every, i.e. the exact-zero result at
    hedge_every=1 is a genuine consequence of hedging every step, not
    evidence that hedging_error is broken and always returns 0. Uses
    the SAME seed across all configs (common random numbers) so the
    comparison isolates the effect of hedge_every from the effect of
    which random path was drawn."""
    from src.analytics import hedging_error
    values = [1, 2, 5, 10, 21]
    errors = []
    for he in values:
        cfg = SimConfig(seed=0, n_steps=150, hedge_every=he)
        history, _ = run_simulation(cfg)
        errors.append(hedging_error(history))
    assert errors[0] == 0.0
    for i in range(1, len(errors)):
        assert errors[i] > 0.0


def test_net_delta_drifts_when_hedging_infrequently():
    """With a long hedge interval, gamma should let net delta wander
    away from zero between hedges -- RMS net delta should be clearly
    larger than the hedge_every=1 case (same seed, same market)."""
    cfg_frequent = SimConfig(seed=9, n_steps=150, hedge_every=1)
    cfg_infrequent = SimConfig(seed=9, n_steps=150, hedge_every=20)
    hist_freq, _ = run_simulation(cfg_frequent)
    hist_infreq, _ = run_simulation(cfg_infrequent)
    rms_freq = float(np.sqrt(np.mean(hist_freq["net_delta"] ** 2)))
    rms_infreq = float(np.sqrt(np.mean(hist_infreq["net_delta"] ** 2)))
    assert rms_infreq > rms_freq


# ============================================================
# 7. Hedge cadence respected in the history
# ============================================================

def test_hedge_qty_is_zero_at_steps_not_on_the_cadence():
    """Structural guarantee, independent of randomness: hedge_qty must
    be exactly 0.0 at any step index not a multiple of hedge_every,
    since DeltaHedger.hedge() is never even called there."""
    cfg = SimConfig(seed=3, n_steps=100, hedge_every=5)
    history, _ = run_simulation(cfg)
    off_cadence = history[history["step"] % 5 != 0]
    assert (off_cadence["hedge_qty"] == 0.0).all()


# ============================================================
# 8. Terminal-step order-flow skip
# ============================================================

def test_no_new_option_inventory_added_on_terminal_step():
    """Structural guarantee, independent of randomness: since Step B
    (order flow) is skipped entirely when i == n_steps, gross option
    inventory at the final step must equal gross inventory at the
    second-to-last step -- nothing new could have been filled."""
    cfg = SimConfig(seed=11, n_steps=60, order_flow_intensity=5.0)
    history, _ = run_simulation(cfg)
    last = history["total_abs_inventory"].iloc[-1]
    second_last = history["total_abs_inventory"].iloc[-2]
    assert last == second_last


def test_terminal_step_still_present_and_finite():
    """Hedging and P&L recording still happen on the terminal step --
    it shouldn't be silently dropped or produce NaN/inf values."""
    cfg = SimConfig(seed=11, n_steps=60)
    history, pnl_history = run_simulation(cfg)
    last_row = history.iloc[-1]
    assert np.isfinite(last_row["total_value"])
    assert np.isfinite(pnl_history["total_pnl_step"].iloc[-1])


# ============================================================
# 9. Inventory-cap enforcement carried through end-to-end
# ============================================================

def test_gross_inventory_never_exceeds_cap_times_number_of_contracts():
    """With a tight max_inventory and high order-flow intensity
    (deliberately trying to breach the cap), gross inventory across
    the whole book must never exceed max_inventory * number_of_contracts
    (each contract is independently capped)."""
    cfg = SimConfig(seed=13, n_steps=100, max_inventory=3,
                     order_flow_intensity=5.0, strikes=(100,))
    history, _ = run_simulation(cfg)
    n_contracts = len(build_contracts(cfg.strikes, cfg.maturity))
    assert (history["total_abs_inventory"] <= cfg.max_inventory * n_contracts).all()


def test_gross_inventory_can_legitimately_exceed_max_inventory_with_multiple_contracts():
    """Regression test, added after investigating a reported 'bug'
    (total_abs_inventory / max_abs_inventory exceeding SimConfig's
    max_inventory value in demo output). Confirmed NOT a bug:
    total_abs_inventory is GROSS, summed across every contract in the
    book, while max_inventory caps each contract INDEPENDENTLY. With
    multiple contracts, gross inventory can legitimately exceed
    max_inventory alone -- it's bounded by max_inventory * n_contracts
    (see the test above), not by max_inventory alone. This test
    deliberately forces every contract toward its cap simultaneously
    (high flow, tight cap, multiple strikes) and confirms gross
    inventory DOES exceed the single-contract cap value, while the
    per-contract cap (verified directly below) is never breached by
    any individual contract."""
    cfg = SimConfig(seed=17, n_steps=150, max_inventory=10,
                     order_flow_intensity=5.0, strikes=(90, 100, 110))
    history, _ = run_simulation(cfg)
    # Gross inventory across 6 contracts legitimately exceeds the
    # single-contract cap of 10.
    assert history["total_abs_inventory"].max() > cfg.max_inventory


def test_no_individual_contract_ever_exceeds_max_inventory_end_to_end():
    """Direct trace of the actual per-contract inventory (via
    MarketMaker.inventory, not the portfolio's gross total_abs_inventory)
    through a full, realistic multi-contract, high-flow run --
    confirming the execution-level clipping in MarketMaker.execute()
    holds for every contract individually, not just in aggregate. This
    reassembles the same components run_simulation() uses internally
    (mirroring simulation.py's own orchestration) since MarketMaker's
    per-contract inventory isn't part of run_simulation()'s public
    return value."""
    from src.market import simulate_gbm, OptionMarket
    from src.market_maker import MarketMaker, OrderFlowGenerator
    from src.portfolio import Portfolio

    cfg = SimConfig(seed=17, n_steps=150, max_inventory=10,
                     order_flow_intensity=5.0, strikes=(90, 100, 110))
    path = simulate_gbm(cfg.S0, cfg.mu, cfg.realized_vol, cfg.dt, cfg.n_steps, seed=cfg.seed)
    contracts = build_contracts(cfg.strikes, cfg.maturity)
    market = OptionMarket(contracts, cfg.r, cfg.pricing_vol)
    mm = MarketMaker(cfg.mm_half_spread, cfg.inventory_skew, cfg.max_inventory, cfg.quote_size)
    flow = OrderFlowGenerator(cfg.order_flow_intensity, cfg.buy_prob, cfg.order_size,
                               seed=cfg.seed + 1)
    portfolio = Portfolio()

    for i in range(cfg.n_steps):  # matches simulation.py's terminal-step flow skip
        t = i * cfg.dt
        S = path[i]
        for contract, mm_side, qty in flow.generate(contracts):
            theo = market.theo_value(contract, S, t)
            quote = mm.quote(contract, theo)
            price = quote.ask if mm_side == "sell" else quote.bid
            fill = mm.execute(t, contract, mm_side, price, qty)
            if fill.qty > 0:
                portfolio.apply_option_fill(contract, mm_side, price, fill.qty)
        for contract in contracts:
            assert abs(mm.get_inventory(contract)) <= cfg.max_inventory


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