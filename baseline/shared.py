from pathlib import Path
import os

from datasets import concatenate_datasets, load_dataset

try:
    from baseline.utils import counting, mfr, zip_files_flat
except ImportError:
    from utils import counting, mfr, zip_files_flat


DATASET_NAMES = {
    "dev": "weerayut/multilexnorm2026-dev-pub",
    "full": "weerayut/multilexnorm2026-full-pub",
}


def load_local_env():
    env_files = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_file in env_files:
        if not env_file.exists():
            continue
        for raw_line in env_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def load_public_dataset(phase, hf_token=None):
    try:
        return load_dataset(DATASET_NAMES[phase], token=hf_token)
    except Exception as exc:
        msg = str(exc)
        if (
            "not in the authorized list" in msg
            or "Cannot access gated repo" in msg
            or "GatedRepo" in msg
        ):
            raise RuntimeError(
                "The Hugging Face token was loaded, but the associated account "
                f"is not authorized for '{DATASET_NAMES[phase]}'. "
                "Request dataset access on Hugging Face with the same account, "
                "then rerun the command."
            ) from exc
        if "403" in msg or "restricted" in msg:
            raise RuntimeError(
                "The Hugging Face dataset is gated. Request access to "
                f"'{DATASET_NAMES[phase]}' and rerun with --hf-token or HF_TOKEN set."
            ) from exc
        raise


def maybe_filter_lang(dataset, lang):
    if not lang:
        return dataset
    return dataset.filter(lambda row: row["lang"] == lang)


def build_language_counts(train_df):
    counts_by_lang = {}
    for lang in sorted(train_df["lang"].unique()):
        train_lang = train_df.loc[train_df["lang"] == lang]
        records = train_lang.to_dict(orient="records")
        counts_by_lang[lang] = counting(records)
    return counts_by_lang


def predict_with_mfr(train_ds, test_ds):
    train_df = train_ds.to_pandas()
    test_df = test_ds.to_pandas()

    counts_by_lang = build_language_counts(train_df)
    test_df["pred"] = test_df.apply(
        lambda row: mfr(row["raw"], counts_by_lang.get(row["lang"])),
        axis=1,
    )
    return test_df


def write_submission(predictions, phase, output_dir=None):
    target_dir = output_dir or Path("outputs") / f"submission_{phase}"
    target_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = target_dir / "predictions.json"
    predictions.to_json(predictions_path, orient="records")

    zip_path = target_dir.with_suffix(".zip")
    zip_files_flat(str(target_dir), str(zip_path))
    return predictions_path, zip_path


def get_submission_predictions(phase, hf_token=None, lang=None):
    data = load_public_dataset(phase, hf_token)
    train = concatenate_datasets([data["train"], data["validation"]])
    if lang:
        train = maybe_filter_lang(train, lang)
        test = maybe_filter_lang(data["test"], lang)
    else:
        test = data["test"]
    return predict_with_mfr(train, test)
