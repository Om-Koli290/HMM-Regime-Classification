# Market Regime Classification Using Hidden Markov Model (HMM)

This project implements a market regime classification system for the S&P 500 using a Gaussian Hidden Markov Model (HMM). The model classifies daily market regimes into bullish, bearish, or sideways states based on features derived from price and volatility data.

## Project Overview
Data Loading: Download S&P 500 and VIX historical data from Yahoo Finance.

Feature Engineering: Compute log returns, volatility, momentum, and standardized VIX as input features.

Modeling: Train a Gaussian HMM to identify hidden market regimes.

Visualization: Plot price series with regime shading and legends.

Usage: The trained model can be saved and later used to classify market regimes in new data.

## Getting Started

Prerequisites:
Python 3.7+

Install dependencies with:
pip install numpy pandas matplotlib yfinance hmmlearn joblib

Run the full pipeline:
python main.py

This will download data, compute features, train the HMM, and visualize regimes on the S&P 500 price chart.

## How to Use the Model for Future Predictions:
Train and save the model with hmm_model.py.

Load the saved model and preprocess new data features.

Use model.predict(new_features) to classify the regime for new dates.
