import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm


def get_normal_distribution_plot(mu, sigma, sample_size: int = 500) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the fitted normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    ax.plot(x, p, color="red", label="Fitted Normal Distribution")

    # Add labels and legend
    ax.set_xlabel("Average Distance")
    ax.set_ylabel("Density")
    ax.set_title("Normal Distribution Fit to Average Distances")
    ax.legend()

    # Return the figure
    return fig


def kl_divergence_normal(mu1, sigma1, mu2, sigma2):
    """
    Calculate the KL divergence between two normal distributions.

    Parameters:
    mu1, sigma1: Mean and standard deviation of the first normal distribution.
    mu2, sigma2: Mean and standard deviation of the second normal distribution.

    Returns:
    KL divergence D_KL(P || Q).
    """
    if sigma1 == 0 or sigma2 == 0:
        return np.inf
    kl_div = (
        np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2) - 0.5
    )
    return kl_div
