MAP_WIDTH = 150
MAP_HEIGHT = 42

START_SYMBOL = "1"
DESTINATION_SYMBOL = "U"

GYMS = [
    "2", "3", "4", "5", "6", "7", "8", "9",
    "B", "C", "D", "E", "G", "H", "I", "J",
    "K", "L", "N", "O", "P", "Q", "S", "T",
]

TERRAIN_COSTS = {".": 1, "R": 5, "F": 15, "A": 30, "M": 200}

GYM_DIFFICULTIES = {
    "2": 35, "3": 40, "4": 45, "5": 50, "6": 55, "7": 60,
    "8": 65, "9": 70, "B": 75, "C": 80, "D": 85, "E": 90,
    "G": 95, "H": 100, "I": 110, "J": 120, "K": 130,
    "L": 140, "N": 150, "O": 155, "P": 160, "Q": 165,
    "S": 170, "T": 180,
}

POKEMON_POWER = {
    "Pikachu": 1.5,
    "Bulbasaur": 1.4,
    "Rattata": 1.3,
    "Caterpie": 1.2,
    "Weedle": 1.1,
}

INITIAL_ENERGY = 6
SPECIAL_SYMBOLS = set(GYMS) | {START_SYMBOL, DESTINATION_SYMBOL}

SPECIAL_ORDER = (START_SYMBOL, *GYMS, DESTINATION_SYMBOL)
SYMBOL_NUMBERS = {symbol: number for number, symbol in enumerate(SPECIAL_ORDER, 1)}
NUMBER_SYMBOLS = {number: symbol for symbol, number in SYMBOL_NUMBERS.items()}


def symbol_number(symbol: str) -> int | None:
    return SYMBOL_NUMBERS.get(symbol)


def symbol_label(symbol: str) -> str:
    number = SYMBOL_NUMBERS.get(symbol)
    return f"{number:02d}" if number is not None else symbol
