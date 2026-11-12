"""Unit tests for speculators.models.utils config-resolution helpers."""

from types import SimpleNamespace
from typing import cast

import pytest
from transformers import PretrainedConfig

from speculators.models.utils import (
    extract_text_config,
    flatten_rope_parameters,
    resolve_draft_intermediate_size,
)


def _fake_verifier(**fields) -> PretrainedConfig:
    """Lightweight stand-in verifier config (the resolver only reads attributes)."""
    return cast("PretrainedConfig", SimpleNamespace(**fields))


# ---------------------------------------------------------------------------
# extract_text_config
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_extract_text_config_plain_model_unchanged():
    config = _fake_verifier(hidden_size=4096)

    assert extract_text_config(config) is config


@pytest.mark.smoke
def test_extract_text_config_flat_multimodal():
    # Qwen3-VL / Llama-4 style: the LLM config sits behind a top-level
    # text_config attribute.
    text_config = _fake_verifier(hidden_size=2048)
    config = _fake_verifier(text_config=text_config)

    assert extract_text_config(config) is text_config


@pytest.mark.smoke
def test_extract_text_config_omni_thinker():
    # Qwen3-Omni style: the top-level config exposes only thinker/talker/
    # code2wav; the LLM config is nested under thinker_config.text_config.
    text_config = _fake_verifier(hidden_size=2048, num_hidden_layers=48)
    thinker = _fake_verifier(text_config=text_config)
    talker = _fake_verifier()
    config = _fake_verifier(
        thinker_config=thinker,
        talker_config=talker,
        code2wav_config=_fake_verifier(),
    )

    assert extract_text_config(config) is text_config


@pytest.mark.smoke
def test_extract_text_config_thinker_without_text_config():
    # A thinker-level config (model_type qwen3_omni_moe_thinker) unwraps to
    # its text_config as well.
    text_config = _fake_verifier(hidden_size=2048)
    thinker = _fake_verifier(text_config=text_config)

    assert extract_text_config(thinker) is text_config


@pytest.mark.smoke
def test_extract_text_config_omni_without_nested_text_config():
    # Degenerate thinker config with no text_config returns the thinker
    # itself rather than crashing.
    thinker = _fake_verifier(hidden_size=2048)
    config = _fake_verifier(thinker_config=thinker)

    assert extract_text_config(config) is thinker


@pytest.mark.smoke
def test_extract_text_config_none_text_config_is_ignored():
    # A None text_config (present but unset) must not be returned.
    config = _fake_verifier(text_config=None, hidden_size=4096)

    assert extract_text_config(config) is config


# ---------------------------------------------------------------------------
# resolve_draft_intermediate_size
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_resolve_uses_dense_intermediate_size_directly():
    # A dense verifier's intermediate_size is mirrored verbatim, even when a
    # hidden_size is also present (dense takes precedence over the 3x fallback).
    verifier = _fake_verifier(intermediate_size=11008, hidden_size=4096)

    assert resolve_draft_intermediate_size(verifier) == 11008


@pytest.mark.smoke
def test_resolve_moe_falls_back_to_3x_hidden_size():
    # MoE verifier: no dense intermediate_size -> draft uses 3 * hidden_size.
    verifier = _fake_verifier(hidden_size=2048)

    with pytest.warns(UserWarning, match="3 x hidden_size"):
        assert resolve_draft_intermediate_size(verifier) == 6144


@pytest.mark.smoke
def test_resolve_ignores_moe_expert_fields():
    # Expert fields are irrelevant now: with no dense intermediate_size the draft
    # width is purely 3 * hidden_size regardless of the MoE routing config.
    verifier = _fake_verifier(
        hidden_size=1024,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        shared_expert_intermediate_size=2048,
    )

    with pytest.warns(UserWarning, match="3 x hidden_size"):
        assert resolve_draft_intermediate_size(verifier) == 3072


@pytest.mark.smoke
def test_resolve_requires_intermediate_or_hidden_size():
    # Degenerate config with neither field -> explicit error pointing at --draft-config.
    verifier = _fake_verifier()

    with pytest.raises(ValueError, match="--draft-config"):
        resolve_draft_intermediate_size(verifier)


# ---------------------------------------------------------------------------
# flatten_rope_parameters
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_flatten_rope_parameters_nested():
    config = _fake_verifier(
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "yarn", "rope_theta": 1000000.0},
        }
    )
    result = flatten_rope_parameters(config)
    assert result is not config
    assert result.rope_parameters == {"rope_type": "default", "rope_theta": 10000.0}


@pytest.mark.smoke
def test_flatten_rope_parameters_already_flat():
    config = _fake_verifier(
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0}
    )
    result = flatten_rope_parameters(config)
    assert result is config


@pytest.mark.smoke
def test_flatten_rope_parameters_none():
    config = _fake_verifier()
    result = flatten_rope_parameters(config)
    assert result is config
