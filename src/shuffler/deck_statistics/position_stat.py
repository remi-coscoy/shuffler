from concurrent.futures import ThreadPoolExecutor
import numpy as np


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


def calculate_chunk_distance(
    shuffled_mat: np.ndarray, unique_numbers: np.ndarray, start: int, end: int
) -> float:
    # Extract chunk from the matrix
    chunk = shuffled_mat[start:end]
    # Calculate the average distance for this chunk
    return position_stat(chunk, unique_numbers)


def position_distrib(
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
            executor.submit(
                calculate_chunk_distance, shuffled_mat, unique_numbers, start, end
            )
            for start, end in chunk_indices
        ]
        # Collect the results after all processes finish
        avg_distances = [future.result() for future in futures]

    return np.array(avg_distances)
