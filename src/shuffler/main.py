from deck import StandardDeck
from shuffles import random_shuffle, no_shuffle, lazy_shuffle
from deck_statistics import stats_main


def main():
    deck = StandardDeck()
    deck.compute_mode()
    sample_size = 1000
    num_threads = 5
    for num_shuffle in [1, 5, 10, 100, 1000]:
        print(f"**{num_shuffle} shuffles**")
        for shuffle_method in [no_shuffle, lazy_shuffle, random_shuffle]:

            result = stats_main(
                arr=deck.cards,
                shuffle_method=shuffle_method,
                number_of_shuffles=num_shuffle,
                sample_size=sample_size,
                num_threads=num_threads,
            )

            print(f"{shuffle_method.__name__}: {result}")


if __name__ == "__main__":
    main()
