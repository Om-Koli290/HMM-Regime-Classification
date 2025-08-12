# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def plot_price_with_regimes(price_series, regimes, title="Price with Market Regimes", regime_labels=None):
    """
    Plot price time series with background shading to indicate market regimes.

    Parameters:
        price_series (pd.Series): price data indexed by date
        regimes (array-like): regime labels (ints), same length and index as price_series
        title (str): plot title
        regime_labels (dict or list, optional): mapping of regime int -> label string,
            e.g. {0: "Bull", 1: "Bear", 2: "Sideways"}
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot price line
    ax.plot(price_series.index, price_series.values, label='S&P 500 Price', color='black')

    regime_series = pd.Series(regimes, index=price_series.index)
    regime_changes = regime_series.ne(regime_series.shift()).cumsum()

    unique_regimes = np.unique(regimes)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_regimes)))
    color_map = dict(zip(unique_regimes, colors))

    # Shade background per regime segment
    for _, group in regime_series.groupby(regime_changes):
        start = group.index[0]
        end = group.index[-1]
        ax.axvspan(start, end, color=color_map[group.iloc[0]], alpha=0.25)

    # Create legend patches for regimes
    if regime_labels is None:
        # Default labels as Regime 0, Regime 1, etc.
        regime_labels = {r: f"Regime {r}" for r in unique_regimes}
    patches = [Patch(color=color_map[r], label=regime_labels[r]) for r in unique_regimes]

    # Add legends
    ax.legend(handles=[plt.Line2D([], [], color='black', label='S&P 500 Price')] + patches, loc='upper left')

    ax.set_title(title)
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from data_loader import load_sp500_and_vix
    from feature_engineering import compute_features
    from hmm_model import train_hmm

    df_raw = load_sp500_and_vix()
    features = compute_features(df_raw)
    model, regimes = train_hmm(features)

    # Optional: define descriptive labels for regimes
    labels = {0: "0", 1: "1", 2: "2"}

    plot_price_with_regimes(df_raw['SP500_AdjClose'].loc[features.index], regimes, regime_labels=labels)
