# main.py

from data_loader import load_sp500_and_vix
from feature_engineering import compute_features
from hmm_model import train_hmm
from visualization import plot_price_with_regimes
from backtest import backtest_regime_strategy
def main():
    df_raw = load_sp500_and_vix()

    features = compute_features(df_raw)

    model, regimes = train_hmm(features, n_states=3)

    regime_labels = {
        0: "Bullish",
        1: "Sideways",
        2: "Bearish"
    }

    prices = df_raw['SP500_AdjClose'].loc[features.index]
    plot_price_with_regimes(prices, regimes, regime_labels=regime_labels)

    results = backtest_regime_strategy(
        prices,
        regimes,
        bull_label=0,
        bear_label=2,
        side_label=1
    )

    print("\n=== Backtest Performance Metrics ===")
    for k, v in results["metrics"].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Trade Log ===")
    print(results["trades"])

if __name__ == "__main__":
    main()

