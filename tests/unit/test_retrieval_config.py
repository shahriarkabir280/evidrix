import pytest
from pydantic import ValidationError

from app.rag.retrieval import PRESETS_DIR, RetrievalConfig, UnknownPresetError, load_preset


@pytest.mark.parametrize("name", sorted(p.stem for p in PRESETS_DIR.glob("*.toml")))
def test_every_preset_file_is_a_valid_config(name):
    assert isinstance(load_preset(name), RetrievalConfig)


def test_presets_describe_the_three_pipelines_to_compare():
    assert (load_preset("dense").mode, load_preset("dense").rerank) == ("dense", False)
    assert (load_preset("hybrid").mode, load_preset("hybrid").rerank) == ("hybrid", False)
    assert load_preset("hybrid_rerank").rerank is True


@pytest.mark.parametrize("name", ["missing", "../../etc/passwd", "dense.toml", ""])
def test_unknown_or_path_like_preset_names_are_refused(name):
    with pytest.raises(UnknownPresetError):
        load_preset(name)


@pytest.mark.parametrize(
    "fields",
    [
        {"top_k": 0},
        {"top_k": 21},
        {"top_k": 10, "candidate_k": 5},
        {"mode": "semantic"},
        {"rrf_k": 0},
        {"unknown": 1},
    ],
)
def test_invalid_configs_are_rejected(fields):
    with pytest.raises(ValidationError):
        RetrievalConfig(**fields)


def test_config_is_immutable():
    config = RetrievalConfig()

    with pytest.raises(ValidationError):
        config.top_k = 9


def test_only_the_rerank_preset_has_an_abstention_threshold():
    assert load_preset("dense").min_score is None
    assert load_preset("hybrid").min_score is None
    assert load_preset("hybrid_rerank").min_score is not None
