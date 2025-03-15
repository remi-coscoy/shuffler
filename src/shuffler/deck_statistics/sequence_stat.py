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
