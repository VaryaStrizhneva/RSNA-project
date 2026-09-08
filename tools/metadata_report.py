from __future__ import annotations

import argparse

from src.rsna.data.metadata import (
    add_clean_reports,
    build_series_slots,
    get_expert_labeled_rows,
    label_summary,
    load_metadata,
    missing_label_summary,
    series_summary,
    slot_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RSNA knee metadata CSVs.")
    parser.add_argument("--data-root", default="data/raw", help="Folder containing the competition CSV files.")
    args = parser.parse_args()

    bundle = load_metadata(args.data_root)
    train = add_clean_reports(bundle.train)
    expert = get_expert_labeled_rows(train)
    slots = build_series_slots(bundle.train_series, study_ids=train["StudyInstanceUID"])

    print("Files")
    print(f"  root: {bundle.root}")
    print(f"  train: {bundle.train.shape}")
    print(f"  train_series: {bundle.train_series.shape}")
    print(f"  test: {bundle.test.shape}")
    print(f"  test_series: {bundle.test_series.shape}")
    print(f"  sample_submission: {bundle.sample_submission.shape}")

    print("\nExpert-labeled studies")
    print(f"  {len(expert)} / {len(train)}")

    print("\nMissing labels")
    print(missing_label_summary(train).to_string(index=False))

    print("\nLabel summary")
    print(label_summary(train).to_string(index=False))

    print("\nSeries summary")
    summaries = series_summary(bundle.train_series)
    for name, frame in summaries.items():
        print(f"\n{name}")
        print(frame.to_string(index=False))

    print("\nSlot coverage")
    print(slot_coverage(slots).to_string(index=False))

    print("\nReport text")
    lengths = train["Report_clean"].str.len()
    print(f"  non-empty: {(lengths > 0).sum()} / {len(train)}")
    print(f"  mean chars: {lengths.mean():.1f}")
    print(f"  median chars: {lengths.median():.1f}")


if __name__ == "__main__":
    main()
