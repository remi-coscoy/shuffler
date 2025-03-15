import time
from deck import StandardDeck
from shuffles import (
    random_shuffle,
)
from deck_statistics import stats_main


def main():
    deck = StandardDeck()
    deck.compute_mode()
    sample_size = 100000
    for num_shuffle in [1, 10, 100]:
        print(f"**{num_shuffle} shuffles**")
        for shuffle_method in [
            random_shuffle,
        ]:
            start_time = time.time()

            result = stats_main(
                arr=deck.cards,
                shuffle_method=shuffle_method,
                number_of_shuffles=num_shuffle,
                sample_size=sample_size,
            )

            elapsed_time = time.time() - start_time

            print(
                f"The average distance from starting position with the {shuffle_method.__name__} method, {num_shuffle} shuffles and {sample_size} sample size is {result:.2f}. \n "
                f"Time taken: {elapsed_time:.2f} seconds"
            )


if __name__ == "__main__":
    main()
