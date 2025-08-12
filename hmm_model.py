# hmm_model.py
import numpy as np
from hmmlearn import hmm
import joblib  # for saving/loading model


def train_hmm(features, n_states=3, covariance_type='full', random_state=42):
    """
    Train a Gaussian HMM on the input feature matrix.

    Parameters:
        features (pd.DataFrame or np.ndarray): shape (n_samples, n_features)
        n_states (int): number of hidden states (regimes)
        covariance_type (str): covariance type for Gaussian emissions ('full', 'diag', etc.)
        random_state (int): seed for reproducibility

    Returns:
        model: trained HMM model
        hidden_states: array of inferred hidden states (regimes), length n_samples
    """
    # Convert to numpy array if needed
    X = features.values if hasattr(features, 'values') else features

    # Initialize Gaussian HMM
    model = hmm.GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=1000,
                            random_state=random_state)

    # Fit model
    model.fit(X)

    # Decode hidden states with Viterbi
    hidden_states = model.predict(X)

    return model, hidden_states


def save_model(model, filename):
    """Save the trained model to disk."""
    joblib.dump(model, filename)


def load_model(filename):
    """Load a trained model from disk."""
    return joblib.load(filename)


if __name__ == "__main__":
    import pandas as pd
    from data_loader import load_sp500_and_vix
    from feature_engineering import compute_features

    # Load and preprocess data
    df_raw = load_sp500_and_vix()
    features = compute_features(df_raw)

    # Train HMM
    model, states = train_hmm(features, n_states=3)

    # Show last 10 inferred states
    print("Last 10 inferred regimes:", states[-10:])
    print(np.unique(states))  # shows all unique regimes detected
    print(np.bincount(states))  # shows how many days each regime occurred

