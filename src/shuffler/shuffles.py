import numpy as np


def random_shuffle(arr: np.typing.NDArray[np.int32]) -> np.typing.NDArray[np.int32]:
    np.random.shuffle(arr)
    return arr
