"""The tokenizers compared, all loaded from local Hugging Face cache files."""
from __future__ import annotations

import glob
import os

HUB = os.path.expanduser("~/.cache/huggingface/hub")

#: (key, label, hub repo directory, models that use it)
PANEL = [
    ("o200k", "o200k (GPT-4o, GPT-4.1, GPT-5, o-series, gpt-oss)", "models--lmstudio-community--gpt-oss-20b-MLX-8bit", "OpenAI GPT-4o and later"),
    ("cl100k", "cl100k (GPT-4, GPT-3.5)", "models--Xenova--gpt-4", "OpenAI GPT-4 / GPT-3.5"),
    ("llama3", "Llama 3 (128k)", "models--unsloth--Meta-Llama-3.1-8B-Instruct", "Meta Llama 3, 3.1, 3.2, 3.3"),
    ("qwen3", "Qwen (151k)", "models--Qwen--Qwen3-8B", "Qwen2, Qwen2.5, Qwen3"),
    ("qwen35", "Qwen3.5 (248k)", "models--mlx-community--Qwen3.5-9B-4bit", "Qwen3.5"),
    ("gemma3", "Gemma 3 (262k)", "models--google--gemma-3-12b-it", "Google Gemma 3"),
    ("gemma2", "Gemma 2 (256k)", "models--unsloth--gemma-2-9b-it", "Google Gemma 2"),
    ("deepseek", "DeepSeek-V3 / R1 (128k)", "models--deepseek-ai--DeepSeek-V3", "DeepSeek V3, R1"),
    ("tekken", "Mistral Tekken (131k)", "models--mistralai--Mistral-Small-3.1-24B-Instruct-2503", "Mistral Small 3.x, Nemo, Pixtral"),
    ("mistral03", "Mistral v0.3 (32k)", "models--mistralai--Mistral-7B-Instruct-v0.3", "Mistral 7B, Mixtral"),
    ("phi4", "Phi-4 (100k)", "models--microsoft--phi-4", "Microsoft Phi-4"),
    ("olmo2", "OLMo 2 (100k)", "models--allenai--OLMo-2-1124-7B-Instruct", "Ai2 OLMo 2"),
    ("smollm2", "SmolLM2 (49k)", "models--HuggingFaceTB--SmolLM2-1.7B-Instruct", "SmolLM2 / SmolLM3"),
    ("gpt2", "GPT-2 (50k)", "models--gpt2", "GPT-2, GPT-3 (r50k)"),
    ("potion", "potion-multilingual (atlas encoder, 500k)", "models--minishlab--potion-multilingual-128M", "the atlas's own encoder"),
]


def tokenizer_path(repo_dir: str) -> str:
    fs = sorted(glob.glob(f"{HUB}/{repo_dir}/snapshots/*/tokenizer.json"))
    if not fs:
        raise FileNotFoundError(repo_dir)
    return fs[-1]


def load_all():
    from tokenizers import Tokenizer
    out = []
    for key, label, repo, users in PANEL:
        tok = Tokenizer.from_file(tokenizer_path(repo))
        out.append((key, label, users, tok))
    return out
