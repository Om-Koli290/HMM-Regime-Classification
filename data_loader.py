import yfinance as yf
import pandas as pd


def load_sp500_and_vix(start_date="2000-01-01", end_date=None):
    """
    Download S&P 500 adjusted close and VIX index data from Yahoo Finance,
    align on dates, forward-fill missing values, and return a DataFrame.

    Parameters:
        start_date (str): start date in YYYY-MM-DD format
        end_date (str or None): end date in YYYY-MM-DD format or None for today

    Returns:
        pd.DataFrame with columns: ['SP500_AdjClose', 'VIX_Close']
        indexed by Date (DatetimeIndex)
    """
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)[['Close']].rename(
        columns={'Close': 'SP500_AdjClose'})
    vix = yf.download("^VIX", start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'VIX_Close'})

    df = pd.merge(sp500, vix, left_index=True, right_index=True, how='inner')
    df.fillna(method='ffill', inplace=True)
    return df


if __name__ == "__main__":
    df = load_sp500_and_vix()
    print(df.tail())
