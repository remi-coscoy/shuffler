import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm


def get_normal_distribution_plot(
    stat_arr: np.ndarray, mu, sigma, sample_size: int = 500
) -> plt.Figure:
    # Downsample if data is too large
    if len(stat_arr) > sample_size:
        stat_arr = np.random.choice(stat_arr, sample_size, replace=False)
    # Create a new figure for the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the histogram of the average distances
    sns.histplot(
        stat_arr,
        kde=False,
        stat="density",
        bins=30,
        color="blue",
        alpha=0.6,
        label="Average Distances",
        ax=ax,
    )

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
    kl_div = (
        np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2) - 0.5
    )
    return kl_div
