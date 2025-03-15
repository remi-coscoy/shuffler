import numpy as np


def random_shuffle(arr: np.typing.NDArray[np.int32]) -> np.typing.NDArray[np.int32]:
    rand_indices = np.argsort(np.random.rand(*arr.shape), axis=1)
    shuffled_arr = arr[np.arange(arr.shape[0])[:, None], rand_indices]
    return shuffled_arr


def no_shuffle(arr: np.typing.NDArray[np.int32]) -> np.typing.NDArray[np.int32]:
    return arr


def lazy_shuffle(arr: np.typing.NDArray[np.int32]) -> np.typing.NDArray[np.int32]:
    # Generate random indices for each row
    num_rows, num_cols = arr.shape
    idx1 = np.random.randint(0, num_cols, size=num_rows)
    idx2 = np.random.randint(0, num_cols, size=num_rows)

    # Ensure idx1 and idx2 are different for each row
    mask = idx1 == idx2
    while np.any(mask):
        idx2[mask] = np.random.randint(0, num_cols, size=np.sum(mask))
        mask = idx1 == idx2

    # Swap elements at idx1 and idx2 for each row
    arr[np.arange(num_rows), idx1], arr[np.arange(num_rows), idx2] = (
        arr[np.arange(num_rows), idx2],
        arr[np.arange(num_rows), idx1],
    )

    return arr
