from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from shuffler.config.logger import setup_logger
from shuffler.shuffles import random_shuffle

logger = setup_logger(__name__)


@dataclass
class Suit:
    name: str
    order: int


@dataclass
class Rank:
    name: str
    order: int


@dataclass
class Card:
    suit: Suit
    rank: Rank
    order: int

    def __str__(self):
        return f"{self.rank.name} of {self.suit.name}, {self.order}"


class Deck(ABC):
    suits: list[Suit]
    ranks: list[Rank]
    display_cards: list[Card]
    compute_cards: np.typing.NDArray[np.int8]

    def __init__(self):
        self.is_compute = False
        self.display_cards = []
        self.create()
        self.check_order()

    def create(self):
        for suite in self.suits:
            for rank in self.ranks:
                self.display_cards.append(Card(suite, rank, order=len(self.ranks) * suite.order + rank.order))

    def sort(self):
        if self.is_compute:
            self.compute_cards.sort()
        else:
            self.display_cards = sorted(self.display_cards, key=lambda card: card.order)

    def check_order(self):
        self.sort()
        for i, card in enumerate(self.cards[:-1]):
            if self.cards[i + 1].order - card.order != 1:
                raise ValueError(
                    f"Please set consecutive orders for suits and ranks, starting at 0: {card} and {self.cards[i - 1]}"
                )

    def compute_mode(self):
        if not self.is_compute:
            self.compute_cards = np.array([card.order for card in self.display_cards], dtype=np.int8)
            self.is_compute = True

    def display_mode(self):
        if self.is_compute:
            order_map = {card.order: card for card in self.display_cards}
            self.display_cards = [order_map[order] for order in self.compute_cards]
            self.is_compute = False

    @property
    def cards(self):
        if self.is_compute:
            return self.compute_cards
        return self.display_cards

    @cards.setter
    def cards(self, value):
        if self.is_compute:
            self.compute_cards = value
        else:
            self.display_cards = value

    def __str__(self):
        return "\n".join(f"#{i}: {card}" for i, card in enumerate(self.cards))

    def __len__(self):
        return len(self.cards)


def shuffle_deck(
    deck: Deck,
    shuffle_method: Callable[[np.typing.NDArray[np.int32]], np.typing.NDArray[np.int32]],
):
    """
    Wrapper for shuffle functions to keep display and compute mode
    """
    display_flag = not deck.is_compute
    if display_flag:
        deck.compute_mode()
    # Shuffle methods are vectorized for matrixes
    deck.cards = shuffle_method(deck.cards.reshape(1, -1))[0]
    if display_flag:
        return deck.display_mode()
    return deck


class StandardDeck(Deck):
    suits = [
        Suit(name="Club", order=0),
        Suit(name="Diamond", order=1),
        Suit(name="Heart", order=2),
        Suit(name="Spade", order=3),
    ]
    ranks = [
        Rank(name="Ace", order=0),
        Rank(name="2", order=1),
        Rank(name="3", order=2),
        Rank(name="4", order=3),
        Rank(name="5", order=4),
        Rank(name="6", order=5),
        Rank(name="7", order=6),
        Rank(name="8", order=7),
        Rank(name="9", order=8),
        Rank(name="10", order=9),
        Rank(name="Jack", order=10),
        Rank(name="Queen", order=11),
        Rank(name="King", order=12),
    ]


if __name__ == "__main__":
    deck = StandardDeck()
    logger.info("Display Deck:")
    logger.info(deck)
    logger.info(f"\nTotal cards in deck: {len(deck)}")

    deck.compute_mode()
    logger.info("Compute Deck:")
    logger.info(deck)
    logger.info(f"\nTotal cards in deck: {len(deck)}")

    shuffle_deck(deck=deck, shuffle_method=random_shuffle)
    logger.info("Shuffled Compute Deck:")
    logger.info(deck)

    deck.display_mode()
    shuffle_deck(deck=deck, shuffle_method=random_shuffle)

    logger.info("Shuffled Display Deck:")
    logger.info(deck)
