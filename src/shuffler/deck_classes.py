from dataclasses import dataclass, field
from typing import List, Dict
from abc import ABC, abstractmethod


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
    id: str = field(init=False)
    suit: Suit
    rank: Rank

    def __post_init__(self):
        self.id = f"{self.suit.name}_{self.rank.name}"


class Deck(ABC):
    suits: List[Suit]
    ranks: List[Rank]
    cards: List[Card]

    def __init__(self):
        self.cards = []
        self.create_deck()

    def create_deck(self):
        for suite in self.suits:
            for rank in self.ranks:
                self.cards.append(Card(suite, rank))

    def __str__(self):
        return "\n".join(f"#{i}: {card.id}" for i, card in enumerate(self.cards))

    def __len__(self):
        return len(self.cards)


class StandardDeck(Deck):
    suits = [
        Suit(name="Club", order=1),
        Suit(name="Diamond", order=2),
        Suit(name="Heart", order=3),
        Suit(name="Spade", order=4),
    ]
    ranks = [
        Rank(name="Ace", order=1),
        Rank(name="2", order=2),
        Rank(name="3", order=3),
        Rank(name="4", order=4),
        Rank(name="5", order=5),
        Rank(name="6", order=6),
        Rank(name="7", order=7),
        Rank(name="8", order=8),
        Rank(name="9", order=9),
        Rank(name="10", order=10),
        Rank(name="Jack", order=11),
        Rank(name="Queen", order=12),
        Rank(name="King", order=13),
    ]


if __name__ == "__main__":
    deck = StandardDeck()
    print("Initial Deck:")
    print(deck)
    print(f"\nTotal cards in deck: {len(deck)}")
