from typing import Callable
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def stats_main(
    arr: np.ndarray,
    shuffle_method: Callable[[np.ndarray], np.ndarray],
    sample_size: int = 100000,
    number_of_shuffles: int = 1,
    num_threads: int = 5,
):
    # Create multiple copies of the deck
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
        chunk_distances = [future.result() for future in futures]

    # Average the distances from all chunks
    overall_avg_distance = np.mean(chunk_distances)

    return overall_avg_distance


def shuffle_chunk_and_compute_position_stat(
    arr: np.ndarray,
    num_elements_per_chunk: int,
    shuffle_method: Callable[[np.ndarray], np.ndarray],
    number_of_shuffles: int,
):
    chunk = np.tile(arr, (num_elements_per_chunk, 1))
    # Shuffle each row of the chunk independently for a specified number of shuffles
    for _ in range(number_of_shuffles):
        chunk = np.apply_along_axis(shuffle_method, 1, chunk)

    # After shuffling, calculate the average distance for the current chunk
    return position_stat(chunk, arr)


def position_stat(shuffled_mat: np.ndarray, unique_numbers: np.ndarray):
    goal_avg_position = unique_numbers.mean()

    total_distance = 0
    num_elements = 0

    for num in unique_numbers:
        # Find the column indices where this number appears in the matrix
        positions = np.where(shuffled_mat == num)[1]
        avg_position = positions.mean()
        total_distance += np.abs(avg_position - goal_avg_position)
        num_elements += 1

    return total_distance / num_elements
