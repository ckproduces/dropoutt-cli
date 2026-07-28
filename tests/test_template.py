"""Chat template rendering and loss-mask derivation.

These templates are vendored as literals rather than fetched, so the suite runs
offline and so a change in an upstream repo cannot silently change what we
assert.
"""

from __future__ import annotations

import pytest

from dropoutt.chat_template import (
    ChatTemplate,
    TemplateRenderError,
    char_spans_to_token_mask,
    offsets_are_reliable,
)

# Qwen-style ChatML with the Hugging Face `{% generation %}` extension. A stock
# SandboxedEnvironment raises TemplateSyntaxError on this tag, which is why the
# package registers a real Jinja extension for it.
QWEN_GEN = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\n' }}"
    "{%- if message.role == 'assistant' %}"
    "{% generation %}{{- message.content }}{{- '<|im_end|>' }}{% endgeneration %}"
    "{%- else %}{{- message.content + '<|im_end|>' }}{%- endif %}"
    "{{- '\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)

# Mistral-style: no generation tag, and it reaches for raise_exception,
# bos_token and eos_token. Spans have to be recovered by difference.
MISTRAL = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m['role'] == 'user' %}{{ '[INST] ' + m['content'] + ' [/INST]' }}"
    "{% elif m['role'] == 'assistant' %}{{ m['content'] + eos_token }}"
    "{% else %}{{ raise_exception('unsupported role') }}{% endif %}{% endfor %}"
)

# Gemma names the assistant role `model`, not `assistant`.
GEMMA = (
    "{{ bos_token }}{% for m in messages %}"
    "{% set role = 'model' if m['role'] == 'assistant' else m['role'] %}"
    "{{ '<start_of_turn>' + role + '\n' }}"
    "{% if role == 'model' %}{% generation %}{{ m['content'] }}"
    "{{ '<end_of_turn>' }}{% endgeneration %}"
    "{% else %}{{ m['content'] + '<end_of_turn>' }}{% endif %}{{ '\n' }}{% endfor %}"
)

CONVO = [
    {"role": "user", "content": "Merhaba"},
    {"role": "assistant", "content": "Selam!"},
    {"role": "user", "content": "Nasilsin"},
    {"role": "assistant", "content": "Iyiyim."},
]


def test_generation_tag_parses_and_yields_spans():
    tpl = ChatTemplate(source=QWEN_GEN, eos_token="<|im_end|>")
    assert tpl.uses_generation_tag
    out = tpl.render(CONVO)
    assert out.spans_from_tag
    assert [out.text[a:b] for a, b in out.generation_spans] == [
        "Selam!<|im_end|>", "Iyiyim.<|im_end|>"
    ]


def test_sentinels_never_leak_into_rendered_output():
    out = ChatTemplate(source=QWEN_GEN, eos_token="<|im_end|>").render(CONVO)
    assert "\x00" not in out.text and "\x11" not in out.text


def test_generation_tag_span_includes_the_terminator():
    """Convention matters: this template trains the model to emit the stop token."""
    out = ChatTemplate(source=QWEN_GEN, eos_token="<|im_end|>").render(CONVO)
    assert all(out.text[a:b].endswith("<|im_end|>") for a, b in out.generation_spans)


def test_difference_fallback_for_templates_without_the_tag():
    tpl = ChatTemplate(source=MISTRAL, bos_token="<s>", eos_token="</s>")
    assert not tpl.uses_generation_tag
    full, spans = tpl.spans_by_difference(CONVO)
    assert [full[a:b] for a, b in spans] == ["Selam!", "Iyiyim."]


def test_difference_fallback_excludes_the_terminator_by_construction():
    """This is why the stop-token check must not call it a data defect."""
    tpl = ChatTemplate(source=MISTRAL, bos_token="<s>", eos_token="</s>")
    full, spans = tpl.spans_by_difference(CONVO)
    assert not any(full[a:b].endswith("</s>") for a, b in spans)


def test_gemma_model_role_is_handled():
    tpl = ChatTemplate(source=GEMMA, bos_token="<bos>", eos_token="<eos>")
    out = tpl.render(CONVO)
    assert "<start_of_turn>model" in out.text
    assert [out.text[a:b] for a, b in out.generation_spans] == [
        "Selam!<end_of_turn>", "Iyiyim.<end_of_turn>"
    ]


def test_raise_exception_becomes_a_render_error_not_a_crash():
    tpl = ChatTemplate(source=MISTRAL, bos_token="<s>", eos_token="</s>")
    with pytest.raises(TemplateRenderError):
        tpl.render([{"role": "tool", "content": "x"}])


def test_broken_template_fails_at_compile_time():
    with pytest.raises(Exception):
        ChatTemplate(source="{% for m in messages %}{{ m.content }}")


def test_char_spans_map_onto_token_mask():
    offsets = [(0, 3), (3, 7), (7, 12), (12, 15)]
    mask = char_spans_to_token_mask(offsets, [(7, 15)])
    assert mask == [False, False, True, True]


def test_offset_reliability_probe():
    text = "merhaba dunya"
    assert offsets_are_reliable(text, [(0, 7), (7, 13)])
    # Offsets into some other (normalized) string must be rejected, because the
    # loss mask derived from them would be silently wrong.
    assert not offsets_are_reliable(text, [(0, 99)])


def test_loss_mask_end_to_end_with_a_real_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(
        ["<|im_start|>user assistant<|im_end|> Merhaba Selam Nasilsin Iyiyim"] * 40,
        trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>"]),
    )
    tpl = ChatTemplate(source=QWEN_GEN, eos_token="<|im_end|>")
    out = tpl.render(CONVO)
    enc = tok.encode(out.text, add_special_tokens=False)
    mask = char_spans_to_token_mask(list(enc.offsets), out.generation_spans)

    assert any(mask), "some tokens must be trainable"
    assert not all(mask), "prompt tokens must not be trainable"
    trainable = "".join(out.text[a:b] for (a, b), m in zip(enc.offsets, mask) if m)
    assert "Selam" in trainable and "Iyiyim" in trainable
    assert "Merhaba" not in trainable, "user content must never be trainable"
