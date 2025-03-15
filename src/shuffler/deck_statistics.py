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
    mat = np.tile(arr, (sample_size, 1))

    # Split the matrix into chunks for multithreading
    chunk_size = mat.shape[0] // num_threads
    chunks = [mat[i * chunk_size : (i + 1) * chunk_size] for i in range(num_threads)]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for chunk in chunks:
            futures.append(
                executor.submit(
                    shuffle_chunk, chunk, shuffle_method, number_of_shuffles
                )
            )

        # Collect the results after all processes finish
        results = [future.result() for future in futures]

    # Recombine the shuffled chunks back into one matrix
    shuffled_mat = np.vstack(results)

    return average_distance_from_starting_position(mat, shuffled_mat)


def shuffle_chunk(
    chunk: np.ndarray,
    shuffle_method: Callable[[np.ndarray], np.ndarray],
    number_of_shuffles: int,
):
    # Shuffle each row of the chunk independently for a specified number of shuffles
    for _ in range(number_of_shuffles):
        chunk = np.apply_along_axis(shuffle_method, 1, chunk)
    return chunk


def average_distance_from_starting_position(
    og_mat: np.ndarray, shuffled_mat: np.ndarray
):
    """Computes the average displacement of elements in shuffled decks."""
    displacement = np.abs(og_mat - shuffled_mat)
    return displacement.mean()
