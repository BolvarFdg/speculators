"""Unit tests for the DFlash-family draft MRoPE stripping.

DFlash-family drafts (dflash/dflash2/dspark/xpress) run on 1-D logical
positions in both training and vLLM serving, so mrope_section inherited from
MRoPE verifiers (Qwen3-Omni, Qwen3-VL) must be dropped from the draft
transformer_layer_config before the speculator config is built.
"""

from types import SimpleNamespace
from typing import cast

from transformers import PretrainedConfig

from speculators.models.dflash.core import _strip_draft_mrope


def _fake_config(**fields) -> PretrainedConfig:
    return cast("PretrainedConfig", SimpleNamespace(**fields))


def test_mrope_section_stripped_from_rope_parameters():
    # Qwen3-Omni thinker text config: mrope with per-axis sections.
    config = _fake_config(
        rope_parameters={
            "rope_type": "mrope",
            "mrope_section": [16, 28, 4],
            "rope_theta": 1000000.0,
        },
    )

    _strip_draft_mrope(config)

    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 1000000.0,
    }


def test_mrope_section_stripped_from_legacy_rope_scaling():
    config = _fake_config(
        rope_scaling={"rope_type": "mrope", "mrope_section": [16, 28, 4]},
    )

    _strip_draft_mrope(config)

    assert config.rope_scaling == {"rope_type": "default"}


def test_partial_rotary_factor_dropped_with_mrope():
    config = _fake_config(
        rope_parameters={
            "rope_type": "mrope",
            "mrope_section": [8, 14, 2],
            "partial_rotary_factor": 0.5,
            "rope_theta": 500000.0,
        },
    )

    _strip_draft_mrope(config)

    assert config.rope_parameters == {
        "rope_type": "default",
        "rope_theta": 500000.0,
    }


def test_default_rope_untouched():
    config = _fake_config(
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )

    result = _strip_draft_mrope(config)

    assert result is config
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 10000.0}


def test_yarn_rope_untouched():
    # Non-mrope scaling must pass through unchanged.
    config = _fake_config(
        rope_scaling={"rope_type": "yarn", "factor": 4.0},
    )

    _strip_draft_mrope(config)

    assert config.rope_scaling == {"rope_type": "yarn", "factor": 4.0}


def test_no_rope_config_untouched():
    config = _fake_config(hidden_size=2048)

    result = _strip_draft_mrope(config)

    assert result is config
