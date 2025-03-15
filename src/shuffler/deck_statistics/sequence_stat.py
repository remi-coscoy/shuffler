import numpy as np


def sequence_stat(shuffled_mat: np.ndarray):
    diff_mat = np.diff(shuffled_mat, axis=1)
    breaks = np.where(diff_mat != 1, 1, 0)
    split_indices = np.where(breaks, np.arange(1, shuffled_mat.shape[1]), 0)
    sequence_lengths = np.diff(
        np.hstack(
            [
                np.zeros((shuffled_mat.shape[0], 1), dtype=int),
                split_indices,
                np.full((shuffled_mat.shape[0], 1), shuffled_mat.shape[1]),
            ]
        ),
        axis=1,
    )
    counts = np.zeros((shuffled_mat.shape[0], 4), dtype=int)  # Columns for 1, 2, 3, 4+
    for i, row in enumerate(sequence_lengths):
        bincounts = np.bincount(row[row > 0])
        bincounts = np.pad(bincounts, (0, max(0, 5 - len(bincounts))), mode="constant")
        counts[i, :3] = bincounts[1:4]
        counts[i, 3] = np.sum(bincounts[4:])
    average_counts = counts.mean(axis=0)
    average_counts = average_counts / np.sum(average_counts)
    return average_counts
