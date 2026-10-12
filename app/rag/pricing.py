"""Token cost per model, from configs/pricing.toml.

Component 6B · Plan: Part 6, step 3.
"""

import tomllib
from decimal import Decimal
from functools import cache
from pathlib import Path

PRICING_PATH = Path(__file__).resolve().parents[2] / "configs" / "pricing.toml"
MILLION = Decimal(1_000_000)


@cache
def load_prices(path: Path = PRICING_PATH) -> dict[str, tuple[Decimal, Decimal]]:
    """model -> (input, output) USD per million tokens. Decimal: money isn't a float."""
    table = tomllib.loads(path.read_text())
    return {
        model: (Decimal(str(p["input_per_million"])), Decimal(str(p["output_per_million"])))
        for model, p in table.items()
    }


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Decimal | None:
    """None for a model without a price: unknown is not the same as free."""
    prices = load_prices().get(model)
    if prices is None:
        return None
    per_input, per_output = prices
    return (prompt_tokens * per_input + completion_tokens * per_output) / MILLION
