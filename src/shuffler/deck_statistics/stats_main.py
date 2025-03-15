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
from deck_statistics.sequence_stat import sequence_stat


@dataclass
class Results:
    time_taken: float
    position_metric: float
    sequence_metric: float
    kl_div_to_rand: float
    position_distribution: Tuple[float, float]
    position_figure: Optional[plt.Figure]

    def save_figure(self, output_path: Path):
        self.position_figure.savefig(output_path)

    def __str__(self):
        return f"""
        Lower is better
        Time: {self.time_taken:.2f}s
        Position Metric: {self.position_metric:.2f}
        Kullback–Leibler divergence to random: {self.kl_div_to_rand:.2f}
        Sequence metric: {self.sequence_metric:.2f}"""


mean_std_by_sub_size = {50: (1.6956037307692304, 0.17847165490304848)}
random_non_sequence_proportion = 0.9626396153846154


def stats_main(
    arr: np.ndarray,
    shuffle_method: Callable[[np.ndarray], np.ndarray],
    sample_size: int = 100000,
    subsample_size: int = 50,
    number_of_shuffles: int = 1,
    num_cpus: int = 5,
    is_plot=False,
):
    start_time = time.time()
    num_elements_per_chunk = sample_size // num_cpus
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = []
        for _ in range(num_cpus):
            futures.append(
                executor.submit(
                    shuffle_chunk_and_compute_stat,
                    arr,
                    num_elements_per_chunk,
                    shuffle_method,
                    number_of_shuffles,
                )
            )
    # Collect the results after all processes finish
    multithread_results = [future.result() for future in futures]
    position_metrics, sequence_metrics, mus, sigmas = zip(*multithread_results)
    # Average the metrics from all chunks
    sequence_metric = np.mean(sequence_metrics)
    sequence_metric = np.abs(sequence_metric - random_non_sequence_proportion) / (
        random_non_sequence_proportion
    )
    position_metric = np.mean(position_metrics) / 13
    mu = np.mean(mus)
    sigma = np.mean(sigmas)
    kl_div = kl_divergence_normal(
        mu,
        sigma,
        mean_std_by_sub_size[subsample_size][0],
        mean_std_by_sub_size[subsample_size][1],
    )
    if is_plot:
        position_figure = get_normal_distribution_plot(mu=mu, sigma=sigma)
    else:
        position_figure = None
    elapsed_time = time.time() - start_time
    result = Results(
        time_taken=elapsed_time,
        position_metric=position_metric,
        sequence_metric=sequence_metric,
        position_distribution=(mu, sigma),
        position_figure=position_figure,
        kl_div_to_rand=kl_div,
    )

    return result


def shuffle_chunk_and_compute_stat(
    arr: np.ndarray,
    num_elements_per_chunk: int,
    shuffle_method: Callable[[np.ndarray], np.ndarray],
    number_of_shuffles: int,
):
    chunk = np.tile(arr, (num_elements_per_chunk, 1))
    # Shuffle each row of the chunk independently for a specified number of shuffles
    for _ in range(number_of_shuffles):
        chunk = shuffle_method(chunk)

    position_metric = position_stat(chunk, arr)
    sequence_metric = sequence_stat(chunk)
    avg_distance_chunk = position_distrib(chunk, arr)
    mu, sigma = norm.fit(avg_distance_chunk)

    # After shuffling, calculate the average distance for the current chunk
    return position_metric, sequence_metric, mu, sigma
