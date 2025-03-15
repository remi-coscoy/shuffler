from deck import StandardDeck
from shuffles import random_shuffle, no_shuffle, lazy_shuffle
from deck_statistics.stats_main import stats_main
from pathlib import Path


def main():
    figure_path = Path("figures")
    figure_path.mkdir(exist_ok=True)
    deck = StandardDeck()
    deck.compute_mode()
    sample_size = 10_000_000
    num_cpus = 4
    for num_shuffle in [1]:
        print(f"**{num_shuffle} shuffles**")
        for shuffle_method in [lazy_shuffle, random_shuffle, no_shuffle]:

            result = stats_main(
                arr=deck.cards,
                shuffle_method=shuffle_method,
                number_of_shuffles=num_shuffle,
                sample_size=sample_size,
                num_cpus=num_cpus,
            )

            print(f"{shuffle_method.__name__}: {result}")


if __name__ == "__main__":
    main()
