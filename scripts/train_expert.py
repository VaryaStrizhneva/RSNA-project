"""Fit a single-pathology expert on its region of interest.

    python -m scripts.train_expert --experiment expert_lateral_meniscus
    python -m scripts.train_expert --experiment expert_lateral_meniscus --fold 0 --epochs 3

Reads the ROI cache, which is 1.2 GB against the classifier's 36 — the whole training
set sits in memory and a step costs no DICOM.

The score this writes is the AUC of one target and cannot be read against the wide
model's 0.792 on the same target: that number carries the comorbidity reasoning, which
is most of it, and a 48 mm crop has none. The honest comparison is a single-target model
on the wide view, which is a separate run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna.config import TARGETS, Config                              # noqa: E402
from rsna.data.labels import load_confidence_table, silence_of       # noqa: E402
from rsna.expert import ExpertConfig, ExpertNet                      # noqa: E402
from rsna.expert.loop import aucs, fit                                # noqa: E402
from rsna.roi import SPECS, cache                                    # noqa: E402
from rsna.eval.metrics import auc_interval                            # noqa: E402
from rsna.train import assign_folds, build_targets                   # noqa: E402
from rsna.train.folds import report_group                            # noqa: E402

EXPERIMENTS = ROOT / "experiments"
DEFAULT_LABELS = "data/external/pilkwang-rsna-knee-llm-labels/report_labels_v2.csv"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="expert_lateral_meniscus")
    ap.add_argument("--roi-cache", default="/data/mgr/rsna-knee/roi-cache", type=Path)
    ap.add_argument("--data-root", default="data/raw", type=Path)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--holdout-gold", action="store_true",
                    help="keep the 58 expert-labelled studies — and anything sharing "
                         "their report — out of training, and score them separately. "
                         "The weak-label AUC says whether the model agrees with the "
                         "label extractor; this says whether the extractor was worth "
                         "agreeing with.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    spec_file = json.loads((EXPERIMENTS / f"{args.experiment}.json").read_text())
    config = ExpertConfig.from_dict(spec_file.get("config", {}))
    run = dict(spec_file.get("run", {}))
    for key in ("epochs", "batch", "lr"):
        if getattr(args, key) is not None:
            config = config.replace(**{key: getattr(args, key)})
    labels = args.labels or run.get("labels", DEFAULT_LABELS)
    unknown = [t for t in config.targets if t not in TARGETS]
    if unknown:
        raise SystemExit(f"{unknown} are not among the twelve targets")

    groups, studies, total = [], None, 0
    log(f"{args.experiment}: {', '.join(config.targets)}")
    for spec, order in zip(config.specs, config.orders):
        volumes, mask, records = cache.load(args.roi_cache, spec)
        names = [r["study"] for r in records]
        if studies is None:
            studies = names
        elif names != studies:
            raise SystemExit(f"{spec.name} was cut over a different set of studies; "
                             f"rebuild it from the same landmark table")
        note = ""
        if config.one_channel:
            # One series per study, the first of the priority that is present. Lazy:
            # the knee boxes are 19 GB memmaps and `fit` reads a batch at a time.
            volumes, mask, chosen = cache.collapse(volumes, mask, order)
            tags = [f"{w}{'-FS' if fs else ''}" for _, w, fs in spec.series]
            counts = [int((chosen == i).sum()) for i in range(len(spec.series))]
            note = ("  one channel from " + ", ".join(
                f"{t} {100 * c / len(chosen):.0f}%"
                for t, c in zip(tags, counts) if c)
                + (f", nothing {100 * (chosen < 0).mean():.1f}%"
                   if (chosen < 0).any() else ""))
        groups.append((volumes, mask))
        total += volumes.nbytes
        log(f"  {spec.name}: {spec.plane} {spec.box_w_mm:.0f}x{spec.box_h_mm:.0f} mm on "
            f"{spec.out_w}x{spec.out_h} px, {spec.slots} slots, "
            f"{100 * mask.any(axis=(1, 2)).mean():.1f} % of studies covered")
        if note:
            log(note)
    log(f"  {len(studies)} studies, {total / 1e9:.2f} GB in all")

    # The twelve-target machinery builds all twelve, then one column is taken. Cheaper
    # than a second code path, and it keeps the weighting identical to the wide model's
    # so a later comparison differs only in the input.
    base = Config(weights=run.get("weights", "uniform"))
    if base.weights == "assertedness":
        base = base.replace(silence=silence_of(labels))
    # Say which weighting ran. It is recorded in history.json either way, but a number
    # quoted from a log should be checkable from that log: the difference between
    # uniform and assertedness was worth +0.0125 on the lateral meniscus, which is the
    # size of the differences these runs are being compared on.
    log(f"  weighting: {base.weights}"
        + (f", silence from {Path(labels).name}" if base.weights == "assertedness" else "")
        + (", gold studies held out" if run.get("holdout_gold") else ""))
    train_csv = pd.read_csv(args.data_root / "train.csv", dtype={"StudyInstanceUID": str})
    derived = pd.read_csv(labels, dtype={"StudyInstanceUID": str}).set_index(
        "StudyInstanceUID")
    confidence = load_confidence_table(labels) if base.weights == "confidence" else None
    y_all, w_all = build_targets(studies, train_csv, derived, base, confidence)

    cols = [TARGETS.index(t) for t in config.targets]
    y, w = y_all[:, cols], w_all[:, cols]
    # A study trains on the targets it has a weight for; one covered by none is dropped.
    covered = (w > 0).any(axis=1)
    for j, t in enumerate(config.targets):
        has = w[:, j] > 0
        log(f"  {t}: {int((y[has, j] > 0.5).sum())} positive of {int(has.sum())} labelled")
    log(f"  {int((~covered).sum())} studies labelled for no target at all, dropped")

    folds = assign_folds(train_csv, base).reindex(studies).fillna(-1).astype(int).values

    gold_index, gold_y, excluded = np.array([], int), None, np.zeros(len(studies), bool)
    holdout_gold = args.holdout_gold or bool(run.get("holdout_gold"))
    # The expert readings, kept whether or not they are held out. With `holdout_gold`
    # they are an external set every fold scores and the five are averaged; without it
    # they are ordinary training studies — carrying a radiologist's label at
    # `gold_weight` rather than the extractor's — and the honest way to score them is the
    # one every other study already gets: by the single fold that did not train on it.
    #
    # The two numbers are **not interchangeable**. Five models averaged beat one, so the
    # out-of-fold figure sits below the held-out one for a reason that has nothing to do
    # with the model. Comparing a run of one kind against a run of the other reads that
    # gap as quality.
    _all_gold = train_csv.set_index("StudyInstanceUID")[TARGETS]
    _all_gold = _all_gold[_all_gold.notna().all(axis=1)]
    gold_truth = {s: row for s, row in
                  zip(_all_gold.index,
                      (_all_gold[list(config.targets)].to_numpy(float) > 0.5).astype(int))}
    if holdout_gold:
        gold = train_csv.set_index("StudyInstanceUID")[TARGETS]
        gold = gold[gold.notna().all(axis=1)]
        at = {s: i for i, s in enumerate(studies)}
        gold_index = np.array(sorted(at[s] for s in gold.index if s in at), int)
        # Also drop anything sharing a gold study's report: identical reports produce
        # identical weak labels, so leaving them in leaks the answer without copying it.
        report_groups = train_csv.set_index("StudyInstanceUID")["Report"].map(report_group)
        shared = set(report_groups.reindex(gold.index).dropna())
        for s_, g in report_groups.items():
            if g in shared and s_ in at:
                excluded[at[s_]] = True
        gold_y = (gold.loc[[studies[i] for i in gold_index], list(config.targets)]
                  .to_numpy(float) > 0.5).astype(int)
        log(f"  gold holdout: {len(gold_index)} expert-labelled studies "
            f"({' / '.join(str(int(c)) for c in gold_y.sum(axis=0))} positive), plus "
            f"{int(excluded.sum()) - len(gold_index)} sharing their report, out of "
            f"training")
        lo, hi = auc_interval(0.85, int(gold_y[:, 0].sum()), int((gold_y[:, 0] == 0).sum()))
        log(f"  an AUC of 0.85 on those would read [{lo:.2f}, {hi:.2f}] — wide enough "
            f"that only a collapse is readable, not a difference")
    out = args.out or Path("out") / args.experiment
    out.mkdir(parents=True, exist_ok=True)

    history, scores, truth, seen, gold_scores = {}, [], [], [], []
    ema_scores: list = []
    oof_gold: list = []
    for f in range(config.folds):
        if args.fold is not None and f != args.fold:
            continue
        val = np.flatnonzero((folds == f) & covered & ~excluded)
        train = np.flatnonzero((folds != f) & (folds >= 0) & covered & ~excluded)
        log(f"fold {f}: {len(train)} train, {len(val)} held out")
        model = ExpertNet(config)
        result = fit(model, groups, y, w, studies, train, val, config,
                     device=args.device, seed=config.seed,
                     gold_index=gold_index, gold_y=gold_y, log=log)
        torch.save({"config": config.to_dict(),
                    "specs": [sp.to_dict() for sp in config.specs], "fold": f,
                    "state": result.state,
                    # Beside the weights, never instead of them: `load_experts` reads
                    # `state`. Preferring the average later is then a reload, not a run.
                    "ema_state": result.ema_state}, out / f"fold{f}.pt")
        history[str(f)] = [vars(e) for e in result.history]
        ema_scores.append(result.ema_scores)
        scores.append(result.scores)
        truth.append(result.truth)
        seen += result.studies
        if result.gold_scores is not None:
            gold_scores.append(result.gold_scores)
        if not holdout_gold:
            for position, study in enumerate(result.studies):
                if study in gold_truth:
                    oof_gold.append((gold_truth[study], result.scores[position],
                                     None if result.ema_scores is None
                                     else result.ema_scores[position]))
        log(f"fold {f} kept: epoch {result.best.epoch}, auc {result.best.auc:.4f} "
            f"[{' '.join(f'{x:.3f}' for x in result.best.per_target)}]"
            + (f", gold {result.best.gold_auc:.4f}"
               if np.isfinite(result.best.gold_auc) else ""))

    p, t = np.concatenate(scores), np.concatenate(truth)
    per = aucs(t, p)
    # Which of the studies scored actually had pixels in any of the ROI's series. A
    # study with none still gets a score — every window is masked, the attention falls
    # back to zero and the head emits its bias — so they arrive as one block of ties and
    # drag the AUC toward 0.5 in proportion to how many of them there are. Measured on
    # the patellofemoral run, 1.5 % of studies uncovered cost 0.0032; a region of
    # interest read from scarcer series costs correspondingly more.
    #
    # Both numbers are reported because they answer different questions: the first is
    # what this branch alone would score on the whole corpus, the second is what it does
    # where it can see anything at all.
    has_pixels = np.zeros(len(studies), bool)
    for _, m in groups:
        has_pixels |= m.any(axis=(1, 2))
    seen_index = {s: i for i, s in enumerate(studies)}
    covered_scored = np.array([bool(has_pixels[seen_index[s]]) for s in seen])

    if any(e is not None for e in ema_scores):
        ep = np.concatenate([e for e in ema_scores if e is not None])
        if len(ep) == len(t):
            eper = aucs(t, ep)
            log(f"\nthe moving average of the weights, over the same studies:")
            for j, name in enumerate(config.targets):
                log(f"  {name:20s} auc {eper[j]:.4f}  "
                    f"({eper[j] - aucs(t, p)[j]:+.4f} against the weights kept)")

    log(f"\nOUT OF FOLD over {len(t)} studies")
    for j, name in enumerate(config.targets):
        pos = int((t[:, j] > 0.5).sum())
        lo, hi = auc_interval(float(per[j]), pos, len(t) - pos)
        log(f"  {name:20s} auc {per[j]:.4f}  [{lo:.3f}, {hi:.3f}]  ({pos} positive)")
    log(f"  {'mean':20s}     {np.nanmean(per):.4f}")
    if not covered_scored.all():
        sub = aucs(t[covered_scored], p[covered_scored])
        log(f"  over the {int(covered_scored.sum())} with pixels in this ROI "
            f"({100 * covered_scored.mean():.1f} %), ignoring the "
            f"{int((~covered_scored).sum())} that have none and all score the prior:")
        for j, name in enumerate(config.targets):
            log(f"  {name:20s} auc {sub[j]:.4f}  ({sub[j] - per[j]:+.4f})")
    log("  that is agreement with the label extractor, which is itself worth 0.841 "
        "against the 58 expert studies with 10 % UNK")
    gold_line = {}
    if oof_gold:
        gy = np.stack([a for a, _, _ in oof_gold])
        gp = np.stack([b for _, b, _ in oof_gold])
        gper = aucs(gy, gp)
        log(f"\nagainst the {len(gy)} expert readings, each scored by the one fold that "
            f"did not train on it:")
        for j, name in enumerate(config.targets):
            pos = int(gy[:, j].sum())
            lo, hi = auc_interval(float(gper[j]), pos, len(gy) - pos)
            log(f"  {name:20s} auc {gper[j]:.4f}  [{lo:.3f}, {hi:.3f}]  ({pos} positive)")
        log("  not comparable with a held-out gold number: that one averages five "
            "models, this one is a single fold per study")
        gold_line = {"per_target": {n: float(v) for n, v in zip(config.targets, gper)},
                     "mean": float(np.nanmean(gper)), "n": int(len(gy)),
                     "protocol": "out-of-fold, gold in training"}
        if all(c is not None for _, _, c in oof_gold):
            ge = aucs(gy, np.stack([c for _, _, c in oof_gold]))
            log(f"  the moving average, same studies: "
                + " ".join(f"{n} {v:.4f} ({v - gper[j]:+.4f})"
                           for j, (n, v) in enumerate(zip(config.targets, ge))))
            gold_line["ema_per_target"] = {n: float(v)
                                           for n, v in zip(config.targets, ge)}
    if gold_scores:
        gp = np.mean(gold_scores, axis=0)
        gper = aucs(gold_y, gp)
        gold_line = {"per_target": {n: float(v) for n, v in zip(config.targets, gper)},
                     "mean": float(np.nanmean(gper)), "n": int(len(gold_y))}
        log(f"\n  against the {len(gold_y)} expert readings, folds averaged:")
        for j, name in enumerate(config.targets):
            pos = int(gold_y[:, j].sum())
            lo, hi = auc_interval(float(gper[j]), pos, len(gold_y) - pos)
            log(f"  {name:20s} auc {gper[j]:.4f}  [{lo:.2f}, {hi:.2f}]  ({pos} positive)")

    (out / "history.json").write_text(json.dumps(
        {"experiment": args.experiment, "config": config.to_dict(),
         "specs": [sp.to_dict() for sp in config.specs], "labels": labels,
         "run": run, "history": history,
         "oof": {"per_target": {n: float(v) for n, v in zip(config.targets, per)},
                 "mean": float(np.nanmean(per)), "n": len(t)},
         "gold": gold_line},
        indent=1))
    table = pd.DataFrame({"study": seen})
    for j, name in enumerate(config.targets):
        table[f"true:{name}"] = t[:, j]
        table[f"pred:{name}"] = p[:, j]
    table.to_csv(out / "holdout.csv", index=False)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
