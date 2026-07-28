"""Static registries should expose the expanded supported surface."""

from dropoutt.registry_data import benchmarks, model_info, models


def test_model_registry_contains_current_major_families():
    ids = {item["hf_id"] for item in models()["models"]}

    assert len(ids) >= 46
    assert "openai/gpt-oss-20b" in ids
    assert "Qwen/Qwen3-32B" in ids
    assert "google/gemma-3-27b-it" in ids
    assert model_info("microsoft/Phi-4-mini-instruct")["license"] == "mit"


def test_benchmark_registry_contains_recent_reasoning_and_code_sets():
    items = {item["id"]: item for item in benchmarks()["benchmarks"]}

    assert len(items) >= 27
    assert items["math_500"]["eval_split"] == "test"
    assert items["aime_2025"]["n_eval"] == 30
    assert items["bigcodebench"]["eval_split"] == "v0.1.4"
    assert items["mmlu_redux_2"]["n_eval"] == 5700
