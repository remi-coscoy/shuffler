from deck import StandardDeck
from shuffles import random_shuffle, no_shuffle, lazy_shuffle
from deck_statistics import stats_main
from pathlib import Path


def main():
    figure_path = Path("figures")
    figure_path.mkdir(exist_ok=True)
    deck = StandardDeck()
    deck.compute_mode()
    sample_size = 1000000
    num_threads = 5
    for num_shuffle in [1, 10, 100]:
        print(f"**{num_shuffle} shuffles**")
        for shuffle_method in [lazy_shuffle, random_shuffle]:

            result = stats_main(
                arr=deck.cards,
                shuffle_method=shuffle_method,
                number_of_shuffles=num_shuffle,
                sample_size=sample_size,
                num_threads=num_threads,
            )

            print(f"{shuffle_method.__name__}: {result}")

            result.save_figure(
                figure_path
                / f"{shuffle_method.__name__}_{sample_size}_{num_shuffle}.png"
            )


if __name__ == "__main__":
    main()
