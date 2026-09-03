"""
market_maker.py
================
A simple two-sided options market maker and an exogenous order-flow
generator that "hits" its quotes.

Quoting logic
-------------
    theo         = the market's theoretical (Black-Scholes) value,
                    computed elsewhere (market.py) and passed in as a
                    plain float -- MarketMaker has no dependency on
                    pricing.py at all.
    spread       = theo * half_spread
    skew         = theo * inventory_skew * current_inventory
    quoted bid   = theo - spread - skew
    quoted ask   = theo + spread - skew

A positive (long) inventory shifts BOTH quotes down: the MM becomes a
less attractive buyer (lower bid) and a more attractive seller (lower
ask), nudging order flow to reduce the position. This is a
deliberately simple, single-parameter version of the inventory-skewing
idea behind more elaborate market-making models (e.g.
Avellaneda-Stoikov) -- linear, interpretable, no optimal control.

Inventory limits are hard caps, enforced TWICE, independently:
  1. At quoting time (MarketMaker.quote): once a side is at its cap,
     that side's quoted size is zero, so a well-behaved caller simply
     won't attempt to trade past the limit.
  2. At execution time (MarketMaker.execute): even if a caller ignores
     size=0 and requests a fill anyway, execute() clips the requested
     quantity down to whatever capacity actually remains (possibly 0),
     so inventory can never be pushed outside
     [-max_inventory, +max_inventory] regardless of caller behavior.
quote() is a convenience for well-behaved callers; execute() is the
actual source of truth and enforces the limit unconditionally.

Order flow
----------
Customer orders arrive as a Poisson process, independent of the MM's
own quotes (a simplifying assumption -- this is "flow" a desk absorbs
regardless of how it prices, e.g. hedgers/retail). Each arrival is a
customer buy (MM sells) or customer sell (MM buys) with probability
buy_prob, sized `size` contracts.

OrderFlowGenerator is deliberately unaware of inventory or limits -- it
only proposes candidate orders. Whether an order actually fills, and
for how much, is decided entirely by MarketMaker.execute().
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Quote:
    bid: float
    ask: float
    bid_size: int
    ask_size: int


@dataclass
class Fill:
    t: float
    contract: object
    side: str  # 'buy' or 'sell', from the MM's perspective
    price: float
    qty: int  # ACTUAL executed quantity, after inventory-cap clipping
              # -- may be less than requested, including 0.


class MarketMaker:
    def __init__(self, half_spread: float, inventory_skew: float,
                 max_inventory: int, quote_size: int = 1):
        """
        half_spread:     own half-spread over theo value (fraction, e.g. 0.05)
        inventory_skew:  how strongly quotes shift per unit of inventory
        max_inventory:   hard position cap per contract (+/-), enforced
                          both at quoting time and at execution time
        quote_size:      contracts offered per side, when not capped
        """
        self.half_spread = half_spread
        self.inventory_skew = inventory_skew
        self.max_inventory = max_inventory
        self.quote_size = quote_size
        self.inventory = {}  # contract -> signed qty

    def get_inventory(self, contract) -> int:
        return self.inventory.get(contract, 0)

    def quote(self, contract, theo: float) -> Quote:
        inv = self.get_inventory(contract)
        spread = theo * self.half_spread
        skew = theo * self.inventory_skew * inv

        bid = theo - spread - skew
        ask = theo + spread - skew
        bid = max(bid, 0.0)
        ask = max(ask, bid)  # never cross

        # Buying pushes inventory up toward +max_inventory; selling
        # pushes it down toward -max_inventory. Each side is capped by
        # the bound it moves toward.
        bid_size = self.quote_size if inv < self.max_inventory else 0
        ask_size = self.quote_size if inv > -self.max_inventory else 0
        return Quote(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)

    def execute(self, t: float, contract, side: str, price: float,
                qty: int) -> Fill:
        """Apply a fill to the MM's inventory, clipping qty down to
        whatever capacity remains under max_inventory.

        side='buy'  -> the MM bought (a customer sold into the MM's bid)
        side='sell' -> the MM sold  (a customer bought at the MM's ask)

        This is the authoritative inventory-limit enforcement point:
        regardless of the requested qty, resulting inventory is
        guaranteed to stay within [-max_inventory, +max_inventory]. If
        only partial capacity remains, only that much is executed; if
        no capacity remains, qty=0 is executed (a "fill" of nothing,
        returned rather than raised, so callers can inspect it and
        move on without special-casing an exception).
        """
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        inv = self.get_inventory(contract)
        if side == "buy":
            capacity = self.max_inventory - inv       # room to go longer
        else:
            capacity = inv - (-self.max_inventory)     # room to go shorter

        executed_qty = max(0, min(qty, capacity))
        signed = executed_qty if side == "buy" else -executed_qty
        self.inventory[contract] = inv + signed

        return Fill(t=t, contract=contract, side=side, price=price,
                    qty=executed_qty)


class OrderFlowGenerator:
    """Exogenous, price-insensitive customer order flow: a Poisson
    process of buy/sell arrivals, generated independently per contract
    per timestep."""

    def __init__(self, intensity: float, buy_prob: float = 0.5,
                 size: int = 1, seed=None):
        """
        intensity: expected number of customer orders per contract,
                   per timestep (Poisson lambda)
        buy_prob:  probability an arriving order is a customer buy
                   (i.e. the MM sells) vs. a customer sell (MM buys)
        """
        self.intensity = intensity
        self.buy_prob = buy_prob
        self.size = size
        self.rng = np.random.default_rng(seed)

    def generate(self, contracts):
        """Returns a list of (contract, mm_side, qty) for one timestep.
        mm_side is already expressed from the MM's perspective: a
        customer buy maps to mm_side='sell' (fill at the MM's ask); a
        customer sell maps to mm_side='buy' (fill at the MM's bid)."""
        orders = []
        for c in contracts:
            n_orders = self.rng.poisson(self.intensity)
            for _ in range(n_orders):
                customer_buys = self.rng.random() < self.buy_prob
                mm_side = "sell" if customer_buys else "buy"
                orders.append((c, mm_side, self.size))
        return orders