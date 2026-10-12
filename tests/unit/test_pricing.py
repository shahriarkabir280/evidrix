from decimal import Decimal

from app.rag import pricing


def test_local_models_cost_nothing():
    assert pricing.cost_usd("qwen3-vl:8b-instruct", 1_500, 40) == 0


def test_unknown_model_has_no_cost():
    assert pricing.cost_usd("some-hosted-model", 1_500, 40) is None


def test_cost_is_tokens_times_price_per_million(tmp_path, monkeypatch):
    prices = tmp_path / "pricing.toml"
    prices.write_text('["hosted"]\ninput_per_million = 0.15\noutput_per_million = 0.60\n')
    read_file = pricing.load_prices.__wrapped__  # the uncached function
    monkeypatch.setattr(pricing, "load_prices", lambda: read_file(prices))

    cost = pricing.cost_usd("hosted", 2_000_000, 1_000_000)

    assert cost == Decimal("0.90")
