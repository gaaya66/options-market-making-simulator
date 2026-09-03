from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_research_results(
    hedge_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    """
    Generate the three main research figures for the project.

    Each figure uses aggregated experiment results rather than
    individual simulation paths.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1. Hedge frequency
    # -----------------------------
    hedge = (
        hedge_df.groupby("hedge_every")
        .agg(
            hedging_error_mean=("hedging_error", "mean"),
            max_drawdown_mean=("max_drawdown", "mean"),
        )
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        hedge["hedge_every"],
        hedge["hedging_error_mean"],
        marker="o",
        label="RMS net delta",
    )

    ax1.set_xlabel("Hedge interval (steps)")
    ax1.set_ylabel("RMS net delta")

    ax2 = ax1.twinx()

    ax2.plot(
        hedge["hedge_every"],
        hedge["max_drawdown_mean"],
        marker="s",
        label="Mean maximum drawdown",
    )

    ax2.set_ylabel("Mean maximum drawdown")

    ax1.set_title("Hedge Frequency and Residual Risk")

    fig.tight_layout()
    fig.savefig(
        f"{output_dir}/research_hedge_frequency.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -----------------------------
    # 2. Volatility mismatch
    # -----------------------------
    vol = (
        vol_df.groupby("realized_vol")
        .agg(
            pnl_mean=("final_pnl", "mean"),
            pnl_std=("final_pnl", "std"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        vol["realized_vol"],
        vol["pnl_mean"],
        yerr=vol["pnl_std"],
        marker="o",
        capsize=4,
    )

    ax.axvline(
        0.25,
        linestyle="--",
        label="Pricing volatility = 25%",
    )

    ax.set_xlabel("Realised volatility")
    ax.set_ylabel("Final P&L (mean ± 1 SD)")
    ax.set_title("Volatility Mis-Specification and P&L Dispersion")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        f"{output_dir}/research_volatility_mismatch.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -----------------------------
    # 3. Inventory limits
    # -----------------------------
    inventory = (
        inventory_df.groupby("max_inventory")
        .agg(
            pnl_mean=("final_pnl", "mean"),
            inventory_mean=("mean_abs_inventory", "mean"),
        )
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        inventory["max_inventory"],
        inventory["pnl_mean"],
        marker="o",
        label="Mean final P&L",
    )

    ax1.set_xlabel("Maximum inventory per contract")
    ax1.set_ylabel("Mean final P&L")

    ax2 = ax1.twinx()

    ax2.plot(
        inventory["max_inventory"],
        inventory["inventory_mean"],
        marker="s",
        label="Mean absolute inventory",
    )

    ax2.set_ylabel("Mean absolute inventory")

    ax1.set_title("Inventory Limits and Market-Maker Risk")

    fig.tight_layout()
    fig.savefig(
        f"{output_dir}/research_inventory_limits.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

if __name__ == "__main__":
    hedge_df = pd.read_csv("output/sweep_hedge_every.csv")
    vol_df = pd.read_csv("output/sweep_realized_vol.csv")
    inventory_df = pd.read_csv("output/sweep_max_inventory.csv")

    plot_research_results(
        hedge_df,
        vol_df,
        inventory_df,
    )