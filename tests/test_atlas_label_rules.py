"""Wording rules for hand-written atlas names, and the kinds they carry."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RULES = Path(__file__).resolve().parents[1] / "tools" / "atlas_label_rules.py"
SPEC = importlib.util.spec_from_file_location("atlas_label_rules", RULES)
assert SPEC and SPEC.loader
rules = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rules)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("Churches, castles and medieval monuments", "subject"),
        ("Moth and sea-snail species stubs", "subject"),
        ("Pipe-delimited tables on assorted subjects", "form"),
        ("Mixed short web fragments with no shared subject", "mixed"),
        ("Languages, linguistics and language learning", "subject"),
        ("Italian literature and poetry", "subject"),
    ],
)
def test_honest_names_pass(name, kind):
    errors, warnings = rules.check_name(name, kind)
    assert errors == []
    assert warnings == []


@pytest.mark.parametrize(
    ("name", "kind", "fragment"),
    [
        ("Names and entries starting with T", "subject", "letter"),
        ("Python, PayPal, P-initial places and prompts", "subject", "letter"),
        ("Linux hardening logs, Brazilian court appeals and sentence-rewriting prompts", "mixed", "Mixed"),
        ("Mixed logs, court appeals and SQL snippets", "mixed", "three or more"),
        ("Short web fragments", "mixed", "Mixed"),
        ("Mixed prose about dogs", "subject", "not named"),
        ("Dogs and cats", "topic", "kind"),
        ("", "subject", "blank"),
    ],
)
def test_misleading_names_are_refused(name, kind, fragment):
    errors, _ = rules.check_name(name, kind)
    assert any(fragment in error for error in errors)


def test_atlas_exposes_kinds_only_when_every_cell_has_one():
    import numpy as np

    from dropoutt.atlas.apply import Atlas

    def atlas(meta):
        return Atlas(
            centroids=np.eye(3, dtype=np.float32),
            region_category=np.zeros(3, dtype=np.int32),
            coords=np.zeros((3, 2), dtype=np.float32),
            probe_coef=np.zeros((0, 3), dtype=np.float32),
            probe_intercept=np.zeros(0, dtype=np.float32),
            probe_classes=np.zeros(0, dtype=np.int32),
            meta=meta,
        )

    named = atlas({"region_kinds": ["subject", "form", "mixed"], "l1_labels": ["a"], "l1_kinds": ["mixed"]})
    assert named.region_kinds == ["subject", "form", "mixed"]
    assert named.l1_kinds == ["mixed"]
    assert atlas({"region_kinds": ["subject"]}).region_kinds == []
    assert atlas({}).region_kinds == []


def test_a_language_as_the_medium_is_a_warning_not_a_subject():
    errors, warnings = rules.check_name("Greek general web pages", "form")
    assert errors == []
    assert any("greek" in warning for warning in warnings)
