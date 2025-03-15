import numpy as np


def random_shuffle(arr: np.typing.NDArray[np.int32]) -> np.typing.NDArray[np.int32]:
    np.random.shuffle(arr)
    return arr


def no_shuffle(arr: np.typing.NDArray[np.int32]) -> np.typing.NDArray[np.int32]:
    return arr


def lazy_shuffle(arr: np.typing.NDArray[np.int32]) -> np.typing.NDArray[np.int32]:
    idx1, idx2 = np.random.choice(len(arr), 2, replace=False)
    arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
    return arr
