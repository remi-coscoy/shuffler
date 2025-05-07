from pathlib import Path

from shuffler.deck import StandardDeck
from shuffler.deck_statistics.stats_main import stats_main
from shuffler.shuffles import lazy_shuffle, random_shuffle


def main():
    figure_path = Path("figures")
    figure_path.mkdir(exist_ok=True)
    deck = StandardDeck()
    deck.compute_mode()
    sample_size = 1000000
    num_cpus = 5
    for num_shuffle in [1, 10, 100]:
        print(f"**{num_shuffle} shuffles**")
        for shuffle_method in [random_shuffle, lazy_shuffle]:
            result = stats_main(
                arr=deck.cards,
                shuffle_method=shuffle_method,
                number_of_shuffles=num_shuffle,
                sample_size=sample_size,
                num_cpus=num_cpus,
                is_plot=True,
            )
            result.save_figure(
                figure_path / f"pos_{shuffle_method.__name__}_{sample_size}_samples_{num_shuffle}_shuffles.png",
                figure_path / f"seq_{shuffle_method.__name__}_{sample_size}_samples_{num_shuffle}_shuffles.png",
            )

            print(f"{shuffle_method.__name__}: {result}")


if __name__ == "__main__":
    main()
