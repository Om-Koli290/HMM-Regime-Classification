# backtest.py

import pandas as pd
import numpy as np


def backtest_regime_strategy(prices: pd.Series, regimes: np.ndarray,
                             bull_label=0, bear_label=2, side_label=1):

    trades = []
    position = None
    entry_price = None
    entry_idx = None
    entry_regime = None

    for i in range(1, len(regimes)):
        prev_regime = regimes[i - 1]
        curr_regime = regimes[i]

        if position is None:
            if curr_regime == bull_label:
                position = "long"
                entry_price = float(prices.iloc[i])
                entry_idx = prices.index[i]
                entry_regime = curr_regime
            elif curr_regime == bear_label:
                position = "short"
                entry_price = float(prices.iloc[i])
                entry_idx = prices.index[i]
                entry_regime = curr_regime

        elif curr_regime != entry_regime:
            exit_price = float(prices.iloc[i])
            exit_idx = prices.index[i]

            if position == "long":
                pnl = (exit_price - entry_price) / entry_price
            elif position == "short":
                pnl = (entry_price - exit_price) / entry_price
            pnl = float(pnl)

            trades.append({
                "EntryDate": entry_idx,
                "ExitDate": exit_idx,
                "Position": position,
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "PnL": pnl,
                "Regime": entry_regime
            })

            position = None
            entry_price = None
            entry_idx = None
            entry_regime = None

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        total_return = float((1 + trades_df["PnL"]).prod() - 1)
        avg_return = float(trades_df["PnL"].mean())
        win_rate = float((trades_df["PnL"] > 0).mean())
        sharpe = (trades_df["PnL"].mean() / trades_df["PnL"].std() * np.sqrt(len(trades_df))
                  if trades_df["PnL"].std() != 0 else 0.0)
        cumulative_returns = (1 + trades_df["PnL"]).cumprod()
        max_dd = float((cumulative_returns / cumulative_returns.cummax() - 1).min())
    else:
        total_return, avg_return, win_rate, sharpe, max_dd = 0, 0, 0, 0, 0

    correct = 0
    total = 0
    for _, row in trades_df.iterrows():
        if row["Regime"] == bull_label:
            total += 1
            if row["ExitPrice"] > row["EntryPrice"]:
                correct += 1
        elif row["Regime"] == bear_label:
            total += 1
            if row["ExitPrice"] < row["EntryPrice"]:
                correct += 1
    accuracy = correct / total if total > 0 else 0

    results = {
        "trades": trades_df,
        "metrics": {
            "Total Return": total_return,
            "Average Trade Return": avg_return,
            "Win Rate": win_rate,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Accuracy": accuracy,
            "Number of Trades": len(trades_df)
        }
    }

    return results
