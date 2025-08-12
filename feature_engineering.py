# feature_engineering.py
import pandas as pd
import numpy as np


def compute_features(df, rolling_window=20):
    """
    Given a DataFrame with columns ['SP500_AdjClose', 'VIX_Close'], compute:
      - daily log returns
      - rolling volatility (std of returns)
      - momentum (cumulative return over rolling_window days)
      - standardized VIX (z-score)
    Standardize all features (mean=0, std=1).

    Parameters:
        df (pd.DataFrame): raw data with adjusted close and VIX columns
        rolling_window (int): window size for volatility and momentum

    Returns:
        pd.DataFrame: standardized feature matrix with columns:
                      ['Returns', 'Volatility', 'Momentum', 'VIX']
    """
    features = pd.DataFrame(index=df.index)

    # Calculate daily log returns
    features['Returns'] = np.log(df['SP500_AdjClose'] / df['SP500_AdjClose'].shift(1))

    # Calculate rolling volatility (std dev of returns)
    features['Volatility'] = features['Returns'].rolling(window=rolling_window).std()

    # Calculate momentum (cumulative return over rolling_window)
    features['Momentum'] = df['SP500_AdjClose'].pct_change(periods=rolling_window)

    # VIX (use raw VIX_Close from input df)
    features['VIX'] = df['VIX_Close']

    # Drop initial rows with NaNs due to rolling calculation
    features = features.dropna()

    # Standardize (z-score)
    features = (features - features.mean()) / features.std()

    return features


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sp500_and_vix

    df_raw = load_sp500_and_vix()
    features = compute_features(df_raw)
    print(features.tail())
