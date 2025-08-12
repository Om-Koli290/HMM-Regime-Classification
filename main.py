# main.py

from data_loader import load_sp500_and_vix
from feature_engineering import compute_features
from hmm_model import train_hmm
from visualization import plot_price_with_regimes

def main():
    # Load raw S&P 500 and VIX data
    df_raw = load_sp500_and_vix()

    # Compute features needed for HMM training
    features = compute_features(df_raw)

    # Train the HMM model and get inferred regimes
    model, regimes = train_hmm(features, n_states=3)

    # Optional: define human-readable regime labels (customize as needed)
    regime_labels = {
        0: "Bullish",
        1: "Sideways",
        2: "Bearish"
    }

    # Visualize the price series with regime shading
    plot_price_with_regimes(df_raw['SP500_AdjClose'].loc[features.index], regimes, regime_labels=regime_labels)

if __name__ == "__main__":
    main()
