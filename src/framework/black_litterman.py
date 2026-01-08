
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def calculate_implied_returns(cov_matrix: np.ndarray, weights: np.ndarray, risk_aversion: float) -> np.ndarray:
    """
    Calculate implied expected returns from the market weights (or prior weights).
    Pi = lambda * Sigma * w
    """
    return risk_aversion * cov_matrix @ weights

def black_litterman_posterior(
    sigma: np.ndarray,
    prior_mu: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: Optional[np.ndarray] = None,
    tau: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Black-Litterman posterior expected returns and covariance.
    
    Parameters:
    - sigma: Covariance matrix (N x N)
    - prior_mu: Prior expected returns (N x 1) (e.g. Implied Returns)
    - P: View matrix (K x N)
    - Q: View returns vector (K x 1)
    - omega: View uncertainty matrix (K x K). If None, will be estimated from P * Sigma * P'.
    - tau: Scalar indicating uncertainty of the prior.
    
    Returns:
    - posterior_mu: Posterior expected returns
    - posterior_sigma: Posterior covariance matrix
    """
    
    # If omega is not provided, use the standard heuristic: diag(P * (tau * Sigma) * P')
    if omega is None:
        tau_sigma = tau * sigma
        P_tau_sigma_P_T = P @ tau_sigma @ P.T
        omega = np.diag(np.diag(P_tau_sigma_P_T))
        
    inv_tau_sigma = np.linalg.inv(tau * sigma)
    inv_omega = np.linalg.inv(omega)
    
    # M = (inv(tau*Sigma) + P.T * inv(Omega) * P)^-1
    M = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
    
    # Posterior Mu = M * (inv(tau*Sigma) * prior_mu + P.T * inv(Omega) * Q)
    posterior_mu = M @ (inv_tau_sigma @ prior_mu + P.T @ inv_omega @ Q)
    
    # Posterior Sigma = Sigma + M  (Note: M is the uncertainty of the estimate of Mu)
    # The posterior predictive covariance is Sigma + M
    posterior_sigma = sigma + M
    
    return posterior_mu, posterior_sigma

def get_bl_weights(posterior_mu: np.ndarray, sigma: np.ndarray, risk_aversion: float) -> np.ndarray:
    """
    Calculate unconstrained mean-variance weights based on posterior returns.
    w = (lambda * Sigma)^-1 * mu
    """
    inv_sigma = np.linalg.inv(risk_aversion * sigma)
    weights = inv_sigma @ posterior_mu
    return weights
