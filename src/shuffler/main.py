from deck import StandardDeck
from shuffles import (
    random_shuffle,
    fisher_yates_shuffle,
    overhand_shuffle,
    riffle_shuffle,
)
from deck_statistics import stats_main


def main():
    deck = StandardDeck()
    for num_shuffle in [1, 10, 100]:
        print(f"**{num_shuffle} shuffles**")
        for shuffle_method in [
            random_shuffle,
            fisher_yates_shuffle,
            overhand_shuffle,
            riffle_shuffle,
        ]:
            result = stats_main(
                deck=deck, shuffle_method=shuffle_method, number_of_shuffles=num_shuffle
            )
            print(
                f"The average distance from starting position with the {shuffle_method.__name__} method and {num_shuffle} shuffles is {result:.2f}"
            )


if __name__ == "__main__":
    main()
