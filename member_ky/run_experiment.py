from baseline.shared import load_local_env, load_public_dataset, maybe_filter_lang
from baseline.utils import counting, evaluate, mfr


def main():
    load_local_env()
    data = load_public_dataset("dev")

    train = maybe_filter_lang(data["train"], "ko")
    validation = maybe_filter_lang(data["validation"], "ko")

    counts = counting(train)
    predictions = [mfr(row["raw"], counts) for row in validation]

    evaluate(
        raw=[row["raw"] for row in validation],
        gold=[row["norm"] for row in validation],
        pred=predictions,
    )


if __name__ == "__main__":
    main()
