from concurrent.futures import ThreadPoolExecutor

import numpy as np


def sequence_stat(shuffled_arr: np.ndarray):
    total_number_of_cards = shuffled_arr.size
    number_of_decks = shuffled_arr.shape[0]

    shuffled_arr = np.hstack(
        (shuffled_arr, np.full((number_of_decks, 1), -10, dtype=np.int8))
    )
    shuffled_arr = np.ravel(shuffled_arr)
    # Create the condition array: True if the difference is 1, False otherwise
    shuffled_arr = np.concatenate(
        ([False], np.diff(shuffled_arr) == 1)
    )  # First element is False because no previous item to compare

    shuffled_arr = np.diff(
        np.where(
            np.concatenate(
                ([shuffled_arr[0]], shuffled_arr[:-1] != shuffled_arr[1:], [True])
            )
        )[0]
    )[::2]
    non_sequence = total_number_of_cards - (shuffled_arr + 1).sum()
    non_sequence_proportion = non_sequence / total_number_of_cards
    return non_sequence_proportion


def sequence_distrib(
    shuffled_mat: np.ndarray,
    unique_numbers: np.ndarray,
    subsample_size: int = 50,
) -> np.ndarray:
    # Determine the number of chunks
    num_chunks = shuffled_mat.shape[0] // subsample_size
    chunk_indices = [
        (i * subsample_size, (i + 1) * subsample_size) for i in range(num_chunks)
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(calculate_chunk_sequence, shuffled_mat, start, end)
            for start, end in chunk_indices
        ]
        # Collect the results after all processes finish
        avg_distances = [future.result() for future in futures]

    return np.array(avg_distances)


def calculate_chunk_sequence(shuffled_mat: np.ndarray, start: int, end: int) -> float:
    # Extract chunk from the matrix
    chunk = shuffled_mat[start:end]
    # Calculate the average distance for this chunk
    return sequence_stat(chunk)
