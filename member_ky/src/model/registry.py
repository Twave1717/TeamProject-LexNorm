from __future__ import annotations


AVAILABLE_MODELS = ["rule_based", "public_byt5"]

MODEL_ALIASES = {
    "mfr": "rule_based",
    "rule": "rule_based",
    "rule_based": "rule_based",
    "byt5": "public_byt5",
    "public_byt5": "public_byt5",
    "t5": "public_byt5",
}

MODEL_DESCRIPTIONS = {
    "rule_based": "MFR rule-based baseline trained from the selected train split.",
    "public_byt5": "Public ÚFAL ByT5 checkpoints from MultiLexNorm 2021.",
}


def canonical_model_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key not in MODEL_ALIASES:
        known = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown model '{name}'. Known model names/aliases: {known}")
    return MODEL_ALIASES[key]


def print_available_models() -> None:
    print("Available model runners:")
    for model_name in AVAILABLE_MODELS:
        print(f"- {model_name}: {MODEL_DESCRIPTIONS[model_name]}")
