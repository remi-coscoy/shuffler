import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from deck_statistics.gaussian_utils import (
    get_normal_distribution_plot,
    kl_divergence_normal,
)
from deck_statistics.position_stat import position_distrib, position_stat


@dataclass
class Results:
    time_taken: float
    position_average: float
    kl_div_to_rand: float
    position_distribution: Tuple[float, float]
    position_figure: Optional[plt.Figure]

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
    is_plot=False,
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
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for chunk in chunks:
            futures.append(
                executor.submit(
                    position_distrib,
                    chunk,
                    arr,
                    subsample_size,
                )
            )

        # Collect the results after all processes finish
        avg_distance_chunks = [future.result() for future in futures]

    avg_distance_arr = np.vstack(avg_distance_chunks)
    mu, sigma = norm.fit(avg_distance_arr)
    kl_div = kl_divergence_normal(
        mu,
        sigma,
        mean_std_by_sub_size[subsample_size][0],
        mean_std_by_sub_size[subsample_size][1],
    )
    if is_plot:
        position_figure = get_normal_distribution_plot(
            avg_distance_arr, mu=mu, sigma=sigma
        )
    else:
        position_figure = None
    elapsed_time = time.time() - start_time
    result = Results(
        time_taken=elapsed_time,
        position_average=overall_avg_distance,
        position_distribution=(mu, sigma),
        position_figure=position_figure,
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
