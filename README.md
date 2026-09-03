# Options Market-Making & Delta-Hedging Simulator

A simulation environment for studying the P&L and risk of an options market
maker who quotes European options and hedges the resulting delta exposure
with the underlying.

The project focuses on a few basic questions in options market making:

- How does hedge frequency affect residual delta risk?
- What happens when pricing volatility differs from realised volatility?
- How do inventory limits affect P&L and inventory risk?
- How does order-flow intensity affect spread capture and inventory?
- How does the quoted spread affect P&L?

The objective is not to optimise the strategy for maximum P&L. The experiments
are used to isolate the main risk/reward trade-offs in the model.

---

## Model

The underlying follows a geometric Brownian motion:

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
$$

The process is simulated using the exact discrete-time transition.

European options are priced using Black-Scholes. The pricing module calculates
price, delta, gamma, vega and theta.

The simulation separates **pricing volatility** from **realised volatility**.
The former is used by the market maker to calculate theoretical option values,
while the latter controls the simulated underlying price process.

At each timestep the market maker:

1. calculates the theoretical option value;
2. posts a bid and ask around that value;
3. receives simulated customer orders;
4. accumulates option inventory;
5. calculates the resulting delta exposure;
6. hedges using the underlying at scheduled intervals;
7. records P&L and risk measures.

---

## Market Making

The market maker quotes around the Black-Scholes theoretical value using an
inventory adjustment:

$$
\text{bid} = V - s - kI
$$

$$
\text{ask} = V + s - kI
$$

where:

- $V$ is the theoretical option value;
- $s$ is the half-spread;
- $I$ is current option inventory;
- $k$ controls the inventory skew.

If the market maker is long an option, the quotes move down. This makes
selling more attractive and buying less attractive, pushing inventory back
towards zero.

A maximum inventory is also imposed for each option contract.

Customer order arrivals are generated using a Poisson process. Order flow is
**exogenous**: the probability of a fill does not depend on the quoted
spread.

This is deliberately simplified. The model is intended to study the effects
of inventory and hedging rather than reproduce a real limit order book.

---

## Delta Hedging

The market maker's net delta is

$$
\Delta_{\text{net}}
=
\Delta_{\text{options}} + N_{\text{shares}}.
$$

At each hedge time, the underlying position is adjusted to target

$$
N_{\text{shares}} = -\Delta_{\text{options}},
$$

so that

$$
\Delta_{\text{net}} \approx 0.
$$

The underlying is traded at the simulated bid or ask and each hedge trade
incurs a commission.

Hedging is discrete rather than continuous. Between hedge times, the
underlying price can move and option delta can change, creating residual
exposure.

The simulator therefore records RMS net delta as a measure of hedging error.

Only delta is hedged. Gamma and vega remain exposed.

---

## P&L and Risk

P&L is tracked separately for the option market-making and hedging
components.

The main quantities recorded are:

- final P&L;
- option P&L;
- hedging P&L;
- P&L-based annualised Sharpe ratio;
- maximum drawdown;
- RMS net delta;
- maximum absolute inventory;
- mean absolute inventory;
- gamma exposure;
- vega exposure.

The separate P&L components make it possible to see how spread capture,
hedging and inventory interact.

---

## Experiments

Each experiment changes one parameter while keeping the rest of the
simulation configuration fixed.

Five random seeds are used for each parameter value. The same seeds are
reused across parameter values where possible so that comparisons are less
affected by the particular underlying path or order-flow realisation.

The experiments are intended to investigate behaviour rather than select the
best-performing parameter.

### 1. Hedge Frequency

The hedge interval is varied from every timestep to every 21 timesteps.

| Hedge interval | RMS net delta | Mean max drawdown |
|---:|---:|---:|
| 1 | 0.00 | -51 |
| 2 | 3.05 | -62 |
| 5 | 4.77 | -79 |
| 10 | 5.58 | -100 |
| 21 | 8.88 | -172 |

Less frequent hedging produces greater residual delta exposure and larger
drawdowns.

P&L is not monotonic because more frequent hedging reduces directional risk
but increases the number of underlying transactions and therefore
transaction costs.

![Hedge frequency](output/research_hedge_frequency.png)

---

### 2. Volatility Mis-Specification

Pricing volatility is fixed at 25% while realised volatility is varied.

At 25% realised volatility, the standard deviation of final P&L across the
five seeds is roughly £111. At 50% realised volatility it rises to roughly
£1,114.

The main effect is therefore a large increase in P&L dispersion and downside
risk when realised volatility moves far away from the volatility used for
pricing.

![Volatility mismatch](output/research_volatility_mismatch.png)

---

### 3. Inventory Limits

The maximum inventory per option contract is varied.

| Maximum inventory | Mean absolute inventory | Mean max drawdown | Mean final P&L |
|---:|---:|---:|---:|
| 5 | 16.4 | -8 | £445 |
| 10 | 28.0 | -16 | £484 |
| 25 | 48.5 | -47 | £516 |
| 50 | 50.8 | -51 | £520 |
| 100 | 50.8 | -51 | £520 |

Tighter inventory limits reduce exposure and drawdown, but also restrict the
amount of option inventory the market maker can accumulate.

The 50 and 100 cases are effectively identical because the 100-contract
limit is not binding in these simulations.

![Inventory limits](output/research_inventory_limits.png)

---

### 4. Order-Flow Intensity

Increasing the order arrival rate creates more opportunities to trade and
capture the quoted spread, but also increases inventory exposure.

Mean absolute inventory increases substantially as order-flow intensity is
increased.

Because order flow is exogenous in this model, these results should not be
interpreted as a realistic prediction of how trading volume changes with
market conditions.

---

### 5. Spread Width

Wider spreads produce higher P&L in the simulation because the same
exogenous order-flow process is used regardless of the spread.

This is a controlled sensitivity test rather than an optimal quoting result.
A more realistic market-making model would make fill probability depend on
the quoted spread.

---

## Project Structure

```text
options-market-making-simulator/
│
├── src/
│   ├── pricing.py
│   ├── market.py
│   ├── market_maker.py
│   ├── hedging.py
│   ├── portfolio.py
│   ├── simulation.py
│   ├── analytics.py
│   ├── experiments.py
│   └── research_plots.py
│
├── experiments/
├── results/
├── tests/
│
├── output/
├── run_demo.py
├── requirements.txt
└── README.md