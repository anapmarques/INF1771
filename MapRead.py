from collections import Counter
from pathlib import Path

from StaticValues import (
    DESTINATION_SYMBOL,
    GYMS,
    MAP_HEIGHT,
    MAP_WIDTH,
    SPECIAL_SYMBOLS,
    START_SYMBOL,
    TERRAIN_COSTS,
)


def load_map(path: str | Path) -> list[list[str]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [list(line) for line in lines]


def find_special_positions(grid: list[list[str]]) -> dict[str, tuple[int, int]]:
    return {
        symbol: (row, col)
        for row, line in enumerate(grid)
        for col, symbol in enumerate(line)
        if symbol in SPECIAL_SYMBOLS
    }


def validate_map(grid: list[list[str]], positions: dict[str, tuple[int, int]]) -> None:
    if len(grid) != MAP_HEIGHT:
        raise ValueError(f"Map must have {MAP_HEIGHT} rows; got {len(grid)}")
    wrong = [row for row, line in enumerate(grid) if len(line) != MAP_WIDTH]
    if wrong:
        raise ValueError(f"Rows with a width other than {MAP_WIDTH}: {wrong}")

    counts = Counter(symbol for line in grid for symbol in line if symbol in SPECIAL_SYMBOLS)
    required = [START_SYMBOL, *GYMS, DESTINATION_SYMBOL]
    invalid = {symbol: counts[symbol] for symbol in required if counts[symbol] != 1}
    if invalid:
        raise ValueError(f"Missing or duplicate special symbols: {invalid}")
    if set(positions) != set(required):
        raise ValueError("Incomplete special positions")

    allowed = set(TERRAIN_COSTS) | SPECIAL_SYMBOLS
    unknown = sorted({symbol for line in grid for symbol in line} - allowed)
    if unknown:
        raise ValueError(f"Unknown symbols: {unknown}")
