import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from pathlib import Path


@dataclass
class Results:
    time_taken: float
    position_average: float
    kl_div_to_rand: float
    position_distribution: Tuple[float, float]
    position_figure: plt.Figure

    def save_figure(self, output_path: Path):
        self.position_figure.savefig(output_path)

    def __str__(self):
        return f"Average position: {self.position_average:.2f}, Kullback–Leibler divergence to random: {self.kl_div_to_rand:.2f} ({self.time_taken:.2f}s)"


mean_std_by_sub_size = {50: (1.6956037307692304, 0.17847165490304848)}


def stats_main(
    arr: np.ndarray,
    shuffle_method: Callable[[np.ndarray], np.ndarray],
    sample_size: int = 100000,
    subsample_size: int = 50,
    number_of_shuffles: int = 1,
    num_threads: int = 5,
):
    start_time = time.time()
    num_elements_per_chunk = sample_size // num_threads
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for _ in range(num_threads):
            futures.append(
                executor.submit(
                    shuffle_chunk_and_compute_position_stat,
                    arr,
                    num_elements_per_chunk,
                    shuffle_method,
                    number_of_shuffles,
                )
            )

        # Collect the results after all processes finish
        multithread_results = [future.result() for future in futures]
        chunk_distances, chunks = zip(*multithread_results)

    # Average the distances from all chunks
    overall_avg_distance = np.mean(chunk_distances)
    avg_distance_arr = position_distrib(
        np.vstack(chunks), arr, subsample_size=subsample_size
    )

    mu, sigma = norm.fit(avg_distance_arr)
    kl_div = kl_divergence_normal(
        mu,
        sigma,
        mean_std_by_sub_size[subsample_size][0],
        mean_std_by_sub_size[subsample_size][1],
    )

    elapsed_time = time.time() - start_time
    result = Results(
        time_taken=elapsed_time,
        position_average=overall_avg_distance,
        position_distribution=(mu, sigma),
        position_figure=get_normal_distribution_plot(avg_distance_arr),
        kl_div_to_rand=kl_div,
    )
    return result


def shuffle_chunk_and_compute_position_stat(
    arr: np.ndarray,
    num_elements_per_chunk: int,
    shuffle_method: Callable[[np.ndarray], np.ndarray],
    number_of_shuffles: int,
):
    chunk = np.tile(arr, (num_elements_per_chunk, 1))
    # Shuffle each row of the chunk independently for a specified number of shuffles
    for _ in range(number_of_shuffles):
        chunk = shuffle_method(chunk)

    # After shuffling, calculate the average distance for the current chunk
    return position_stat(chunk, arr), chunk


def position_stat(shuffled_mat: np.ndarray, unique_numbers: np.ndarray) -> float:
    goal_avg_position = unique_numbers.mean()

    # Get the indices of the unique numbers in the shuffled matrix
    positions = np.array([np.where(shuffled_mat == num)[1] for num in unique_numbers])

    # Calculate the average position for each unique number
    avg_positions = positions.mean(axis=1)

    # Calculate the absolute differences from the goal_avg_position
    distances = np.abs(avg_positions - goal_avg_position)

    # Return the average of these distances
    return distances.mean()


def position_distrib(
    shuffled_mat: np.ndarray,
    unique_numbers: np.ndarray,
    subsample_size: int = 50,
    num_threads: int = 500,
) -> np.ndarray:
    # Determine the number of chunks
    num_chunks = shuffled_mat.shape[0] // subsample_size
    chunk_indices = [
        (i * subsample_size, (i + 1) * subsample_size) for i in range(num_chunks)
    ]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                calculate_chunk_distance, shuffled_mat, unique_numbers, start, end
            )
            for start, end in chunk_indices
        ]
        # Collect the results after all processes finish
        avg_distances = [future.result() for future in futures]

    return np.array(avg_distances)


def calculate_chunk_distance(
    shuffled_mat: np.ndarray, unique_numbers: np.ndarray, start: int, end: int
) -> float:
    # Extract chunk from the matrix
    chunk = shuffled_mat[start:end]
    # Calculate the average distance for this chunk
    return position_stat(chunk, unique_numbers)


def get_normal_distribution_plot(stat_arr: np.ndarray) -> plt.Figure:
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
    mu, std = norm.fit(stat_arr)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
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
