import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from deck_statistics.gaussian_utils import (
    get_normal_distribution_plot,
    kl_divergence_normal,
)
from deck_statistics.position_stat import position_distrib, position_stat
from deck_statistics.sequence_stat import sequence_distrib, sequence_stat
from scipy.stats import norm


@dataclass
class Results:
    time_taken: float
    position_metric: float
    kl_div_pos: float
    position_distribution: Tuple[float, float]
    position_figure: Optional[plt.Figure]
    kl_div_seq: float
    sequence_distribution: Tuple[float, float]
    sequence_figure: Optional[plt.Figure]
    sequence_metric: float

    def save_figure(self, output_path_pos: Path, output_path_seq: Path):
        self.position_figure.savefig(output_path_pos)
        self.sequence_figure.savefig(output_path_seq)

    def __str__(self):
        return f"""
        Lower is better
        Time: {self.time_taken:.2f}s
        -- POSITION
        Distance to perfect shuffle (0 to 1): {self.position_metric:.2f}
        Kullback–Leibler divergence to perfect shuffle: {self.kl_div_pos:.2f}
        -- SEQUENCE
        Distance to perfect shuffle (0 to 1): {self.sequence_metric:.2f}
        Kullback–Leibler divergence to perfect shuffle: {self.kl_div_seq:.2f}
        """


mean_std_by_sub_size_pos = {50: (1.6956037307692304, 0.17847165490304848)}
mean_std_by_sub_size_seq = {50: (0.9626701346153845, 0.005285948724381456)}
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
    if (
        subsample_size not in mean_std_by_sub_size_seq
        or subsample_size not in mean_std_by_sub_size_pos
    ):
        raise ValueError(
            "Please compute the mean and sigma of the random shuffle for this subsample first"
        )
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
    position_metrics, sequence_metrics, mus_pos, mus_seq, sigmas_pos, sigmas_seq = zip(
        *multithread_results
    )
    # Average the metrics from all chunks
    sequence_metric = np.mean(sequence_metrics)
    sequence_metric = np.abs(sequence_metric - random_non_sequence_proportion) / (
        random_non_sequence_proportion
    )
    position_metric = np.mean(position_metrics) / 13
    mu_pos = np.mean(mus_pos)
    mu_seq = np.mean(mus_seq)
    sigma_pos = np.mean(sigmas_pos)
    sigma_seq = np.mean(sigmas_seq)
    kl_div_pos = kl_divergence_normal(
        mu_pos,
        sigma_pos,
        mean_std_by_sub_size_pos[subsample_size][0],
        mean_std_by_sub_size_pos[subsample_size][1],
    )
    kl_div_seq = kl_divergence_normal(
        mu_seq,
        sigma_seq,
        mean_std_by_sub_size_seq[subsample_size][0],
        mean_std_by_sub_size_seq[subsample_size][1],
    )
    if is_plot:
        position_figure = get_normal_distribution_plot(mu=mu_pos, sigma=sigma_pos)
        sequence_figure = get_normal_distribution_plot(mu=mu_seq, sigma=sigma_seq)
    else:
        position_figure = None
        sequence_figure = None
    elapsed_time = time.time() - start_time
    result = Results(
        time_taken=elapsed_time,
        position_metric=position_metric,
        position_distribution=(mu_pos, sigma_pos),
        position_figure=position_figure,
        kl_div_pos=kl_div_pos,
        sequence_metric=sequence_metric,
        sequence_distribution=sequence_distrib,
        sequence_figure=sequence_figure,
        kl_div_seq=kl_div_seq,
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
    avg_sequence_chunk = sequence_distrib(chunk, arr)
    mus_pos, sigmas_pos = norm.fit(avg_distance_chunk)
    mus_seq, sigmas_seq = norm.fit(avg_sequence_chunk)

    # After shuffling, calculate the average distance for the current chunk
    return position_metric, sequence_metric, mus_pos, mus_seq, sigmas_pos, sigmas_seq
