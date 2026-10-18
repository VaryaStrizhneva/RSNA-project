"""The whole submission when experts are in it: landmarks, regions, seven branches.

`run_submission` reads one wide package over one decoded cache and is done. An expert
submission has a dependency the wide one does not: **where to cut is itself predicted**,
so the landmark models must run before any region can be read, and the regions must be
cut before any expert can score.

    wide package ─────────────────────────────┐
    headers → landmark models → regions → experts → columns they own
                                               └──→ submission.csv

Three decisions are worth stating, because each is a failure this would otherwise have.

**The wide model goes first and owns every column.** Experts then overwrite only the
columns they own, and only for the studies they actually scored. A study whose landmark
could not be predicted, whose series the region needs is missing, **or that was cut over
no pixels at all**, keeps the wide model's answer instead of a prior — and keeps a row,
which the submission format requires whatever happened upstream.

That last case is not hypothetical and it is not rare: 299 of 4407 studies reach an
expert as an all-zero tensor, because they carry no series matching any of its regions.
The model answers anyway, with a constant, so writing it in would swap a real ordering
for a tie. Abstaining is the whole reason the wide model runs first.

**One region at a time.** Cutting all seven at once is 10.3 GB for 1300 studies; cutting
one, scoring it and dropping it is 1.6 GB at the peak. The scored notebook has 20 GB of
writable disk and other things to put in it.

**Only files this function wrote are deleted.** Pointed at the training cache directory
it will reuse what is already cut rather than redo it, and will not remove it. A rehearsal
that deletes the caches it was asked to verify against is a bad afternoon.
"""

from __future__ import annotations

from collections.abc import Mapping

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import TARGETS, Config
from ..dicom import annotate, walk
from ..roi import cache
from ..expert.config import ExpertConfig
from .experts import (JITTER_VIEWS, inplane_jitter, load_experts,
                      mirror_groups, score_experts)
from .landmarks import load_landmarks, predict_landmarks
from .run import run_submission


def _headers(data_root: Path, split: str) -> pd.DataFrame:
    """The annotated header table, with the competition's own plane label attached."""

    headers = annotate(walk(data_root, split))
    if headers.empty:
        raise FileNotFoundError(f"no series under {data_root / split}")
    csv = data_root / f"{split.replace('_series', '')}_series.csv"
    planes = pd.read_csv(csv)
    headers["plane"] = headers["SeriesInstanceUID"].map(
        dict(zip(planes["SeriesInstanceUID"], planes["Anatomical_Plane"])))
    return headers


def _points_needed(expert_runs, device: str, log=print) -> tuple[dict, set]:
    """What each expert is, and which landmarks the whole set depends on."""

    loaded, wanted = {}, set()
    for run in expert_runs:
        models, config, specs = load_experts(run, device, log=log)
        for target in config.targets:
            if target not in TARGETS:
                raise ValueError(f"{run} claims {target!r}, not one of the twelve")
        loaded[str(run)] = (models, config, specs)
        for spec in specs:
            wanted.add(spec.landmark)
            if spec.landmark2:
                wanted.add(spec.landmark2)
    # **Several runs may claim the same column, and that is the point.** Until
    # 2026-10-18 this raised, because two runs writing the same column one after the
    # other would have kept whichever ran last. They are now averaged instead, which is
    # a decision and a measured one: over the eight columns, the shipped run plus the
    # best other run of that column is worth **+0.0083 of AUC a column out of fold, 39
    # paired folds of 40**, and the gain is not an accident of which run is better --
    # the partner is the *weaker* model on seven of the eight, and the pair still wins.
    #
    # The arithmetic, on ACL. An AUC counts (positive, negative) pairs. The two runs
    # agree on 87.6 % of them -- both right on 81.0 %, both wrong on 6.5 % -- and
    # averaging can do nothing there. On the 12.4 % where they disagree the shipped run
    # alone is right 47.8 % of the time and their average is right 62.4 %, so
    # `0.124 x (0.624 - 0.478) = +0.018`, against +0.0195 measured. The mechanism is
    # entirely in that line: averaging keeps the *magnitude* of each disagreement, so
    # the confident reading wins the pair, and an error specific to one crop or one
    # encoder is diluted while the signal both runs share is not.
    owners: dict = {}
    for run, (_, config, _) in loaded.items():
        for target in config.targets:
            owners.setdefault(target, []).append(run)
    for target, runs in sorted(owners.items()):
        if len(runs) > 1:
            log(f"{target}: {len(runs)} runs, averaged as ranks")
    return loaded, wanted


def run_expert_submission(expert_runs, landmark_runs, package, data_root,
                          split: str = "test_series", encoder=None,
                          out="submission.csv", scratch=None, device: str = "cpu",
                          workers: int = 8, keep_cache: bool = False,
                          blend: float | Mapping[str, float] | None = None,
                          mirror_tta: bool = True,
                          jitter_tta: bool = False,
                          log=print) -> Path:
    """Write a submission whose expert columns come from experts. Returns the path.

    `blend` is the weight the expert carries against the wide model, both as ranks:
    `None` hands the column to the expert outright, `0.5` averages the two. Measured
    out of fold over 4348 studies, averaging is worth **+0.0259** of macro against the
    wide model alone while the expert alone is worth +0.0159 — the wide model keeps
    saying something the expert does not, on every one of the seven targets.

    Tuning the weight per target is not worth it. Swept over a grid and chosen on four
    folds, the best weights land between 0.51 and 0.72 and buy **+0.0010** over a flat
    half — a twenty-fifth of what blending at all buys, and under the ~0.005 that the
    public leaderboard can resolve at all.

    `mirror_tta` reads every sagittal region a second time with its depth axis reversed
    and averages the two. On by default: the axis is a nuisance the folds have already
    seen both signs of — `mirror_groups` carries the measurement — so the second reading
    costs one more forward pass and no training at all. It is a flag rather than
    unconditional because a submission with it and one without are the only way to see
    what it is worth where it is being paid for.

    `jitter_tta` does the same over the in-plane jitter the folds trained under,
    on every plane rather than only the sagittal ones. **Off by default**: it costs
    one forward pass per view where the mirror costs one in total, and at roughly
    14 minutes a pass over the eight experts on 2xT4 five extra views is an hour and
    a half out of whatever the host leaves of the nine.
    """

    if data_root is None:
        raise ValueError("data_root is None — the caller did not resolve the competition "
                         "root, and pathlib's error for that is three frames deep")
    data_root, out = Path(data_root), Path(out)
    expert_runs = [Path(r) for r in expert_runs]

    # An expert whose every column the mapping leaves to the host is dropped **here**,
    # before anything is loaded. `_overwrite` already skips such a column, but it does so
    # at the end: the cache would have been cut and the five folds run first, and thrown
    # away after. That cost nothing while every column was named; it costs eight caches,
    # eight inferences and three landmark models to use one expert as soon as a probe
    # names a single column, which is exactly what the per-column measurements do.
    #
    # Dropping them here also shrinks `wanted`, so only the landmark models those experts
    # actually hang off are run. Nothing changes for an ordinary submission, where the
    # mapping names all eight.
    if isinstance(blend, Mapping):
        asked = {t for t, w in blend.items() if w is not None}
        keep, drop = [], []
        for run in expert_runs:
            ck = torch.load(next(Path(run).glob("fold*.pt")), map_location="cpu",
                            weights_only=False)
            targets = set(ExpertConfig.from_dict(ck["config"]).targets)
            (keep if targets & asked else drop).append(run)
        if drop:
            log(f"{len(drop)} expert(s) own no column this blend names and are not "
                f"loaded: {', '.join(r.name for r in drop)}")
        if not keep:
            raise ValueError(f"no expert owns any of {sorted(asked)}; nothing to graft")
        expert_runs = keep
    landmark_runs = [Path(r) for r in landmark_runs]

    loaded, wanted = _points_needed(expert_runs, device, log=log)
    log(f"{len(expert_runs)} expert run(s) over {len(wanted)} landmark(s): "
        f"{', '.join(sorted(wanted))}")

    # The wide model first: it writes the benchmark fallback before anything expensive,
    # and its twelve columns are the floor every expert improves on rather than replaces.
    #
    # `package=None` says that floor already exists at `out` — someone else's pipeline
    # wrote it, and the experts are grafting onto it rather than onto ours. The whole
    # machinery below is indifferent to which: it reads twelve columns of ranks and
    # replaces the seven it owns, so the host can be our five folds of DINOv2 or a
    # thirty-model public ensemble.
    if package is not None:
        run_submission(package, data_root, split, encoder=encoder, out=out,
                       device=device, log=log)
    elif not out.is_file():
        raise FileNotFoundError(
            f"package is None, so {out} was expected to hold the submission the experts "
            f"graft onto — nothing wrote it")
    frame = pd.read_csv(out, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    missing = [t for t in TARGETS if t not in frame.columns]
    if missing:
        raise ValueError(f"{out} does not carry {missing}; it is not a submission")
    log(f"{'wide model' if package is not None else 'host submission'}: "
        f"{len(frame)} studies, {len(frame.columns)} columns")

    headers = _headers(data_root, split)
    log(f"{len(headers)} series over {headers.StudyInstanceUID.nunique()} studies")

    owned = tempfile.mkdtemp(prefix="rsna-roi-") if scratch is None else None
    room = Path(owned or scratch)
    room.mkdir(parents=True, exist_ok=True)
    # tag -> the files that cut wrote for it, so a cut can be dropped by name rather
    # than by parsing one back out of a filename (`lateral_mm:g` can put a '.' in a tag).
    made: dict[str, list[Path]] = {}

    pooled: dict[str, dict[str, list[float]]] = {}
    # How many runs still need each cut. Several runs may read the **same** region --
    # the ACL, MCL and Baker's pairs each do -- and a cut dropped after the first of
    # them is simply cut again for the second. Counted here so a cut survives until its
    # last reader has used it, and no longer: the cuts are the expensive part of the
    # leg, and holding all of them at once is what the deletion was there to avoid.
    remaining: dict[str, int] = {}
    for _run in expert_runs:
        for _spec in loaded[str(_run)][2]:
            tag = cache.cache_tag(_spec)
            remaining[tag] = remaining.get(tag, 0) + 1
    try:
        points = _landmarks(landmark_runs, wanted, headers, device, workers, log)
        table = room / "landmarks.csv"
        points.to_csv(table, index=False)

        for run in expert_runs:
            models, config, specs = loaded[str(run)]
            log(f"{run.name}: {', '.join(config.targets)}")
            groups, names, tags = [], None, []
            for spec in specs:
                tag = cache.cache_tag(spec)
                tags.append(tag)
                if not (room / f"{tag}.json").exists():
                    cache.build(table, headers, room, spec, Config(),
                                workers=workers, log=lambda m: log(f"    {m}"))
                    made[tag] = [room / f"{tag}.npy", room / f"{tag}_mask.npy",
                                 room / f"{tag}.json"]
                else:
                    log(f"    reusing the cut of {spec.name}")
                volumes, mask, records = cache.load(room, spec)
                cut = [r["study"] for r in records]
                if names is None:
                    names = cut
                elif cut != names:
                    raise ValueError(
                        f"{spec.name} was cut over a different set of studies than "
                        f"{specs[0].name}; one expert cannot read two alignments")
                groups.append((volumes, mask))

            scores = score_experts(models, groups, batch=config.batch, device=device)

            # Two test-time averages, each measured on its own out of fold, combined at
            # **equal weight per estimator** rather than by pooling every reading into
            # one mean. Pooling would weight the mirror 1 in 7 instead of 1 in 2 and so
            # dilute the larger of the two effects; equal weights keep each estimator as
            # it was measured, and neither was tuned against the other.
            estimators = []
            sagittal = sum(sp.plane == "Sagittal" for sp in specs)

            # The sagittal depth axis reverses with the knee's side -- `mirror_groups`
            # carries the measurement -- so the reversed reading is an input the folds
            # have already seen. What it buys is an ensemble of two views rather than a
            # better single one: out of fold on `expert_lm_d32_cnxnano` the reversed
            # reading alone scores 0.8955 against 0.8958 while agreeing with itself only
            # at a Spearman of 0.921. Worth +0.0024 a column over 15 paired folds.
            if mirror_tta and sagittal:
                other = score_experts(models, mirror_groups(groups, specs),
                                      batch=config.batch, device=device)
                estimators.append((scores + other) / 2)
                log(f"  averaged with the depth-reversed reading of {sagittal} "
                    f"sagittal region(s)")

            # The in-plane jitter the folds trained under. Covers every plane rather than
            # only the sagittal ones, and is worth +0.0009 a column over eight -- but it
            # costs one forward pass per view, which is why it is off by default.
            if jitter_tta:
                parts = [scores]
                for _name, angle, scale, dx, dy in JITTER_VIEWS:
                    parts.append(score_experts(
                        models, groups, batch=config.batch, device=device,
                        transform=inplane_jitter(angle, scale, dx, dy)))
                estimators.append(np.mean(parts, axis=0))
                log(f"  averaged over {len(JITTER_VIEWS)} jittered readings")

            if estimators:
                scores = np.mean(estimators, axis=0)

            # A study whose every slot is masked was cut over no pixels at all. The
            # model still answers — measured over the corpus it answers with a
            # *constant* (the MCL expert returns 0.2499 to 0.2516, sd 0.0006, for the
            # 197 studies that carry no fat-suppressed coronal) — so writing that
            # column in trades the wide model's ordering for a tie on those rows.
            #
            # 299 of 4407 studies are in this position across the seven shipped
            # experts, 197 of them for MCL and 64 for patellofemoral OA. None is
            # recoverable by relaxing the series rule: they do not carry the sequence,
            # and the slot that would cover them is the non-fat-suppressed one whose
            # removal is worth +0.0117 on MCL. So the expert abstains and the wide
            # model keeps the row, which is what it is there for.
            #
            # Emptiness is judged across *all* of an expert's regions, not each: a
            # study the sagittal spec could not cut but the coronal one could has
            # pixels, and the model saw them.
            seen = has_pixels(groups, len(names))
            if not seen.all():
                log(f"  {int((~seen).sum())} study(ies) carry no pixels for this "
                    f"expert; the wide model keeps those rows")
            # Held per target rather than written now, because another run may own the
            # same column. Each run contributes a **percentile rank over the studies it
            # itself could read**, which is what makes two runs comparable at all: they
            # emit probabilities on different scales, and one may abstain on studies the
            # other reads. A study read by one run only keeps that run's reading, which
            # is the right answer and not a special case.
            kept = [n for n, k in zip(names, seen) if k]
            if not kept:
                continue
            for j, target in enumerate(config.targets):
                pct = pd.Series(scores[seen][:, j], index=kept).rank(pct=True)
                bucket = pooled.setdefault(target, {})
                for study, value in pct.items():
                    bucket.setdefault(study, []).append(float(value))

            for tag in tags:
                remaining[tag] -= 1
            if not keep_cache:
                for tag in [t_ for t_ in made if remaining.get(t_, 0) <= 0]:
                    for path in made.pop(tag):
                        if path.exists():
                            path.unlink()

        # One write per column, over however many runs claimed it. The mean is of
        # **percentile ranks**, not probabilities: two runs trained with different crops
        # and encoders are not calibrated against each other, and the metric is an AUC,
        # so only the ordering carries anything. `_overwrite` then quantile-maps that
        # ordering onto the values the column already holds, exactly as it did for a
        # single run.
        for target in sorted(pooled):
            per_study = pooled[target]
            studies = sorted(per_study)
            vals = np.array([sum(v) / len(v) for v in
                             (per_study[s] for s in studies)], float)
            counts = {len(per_study[s]) for s in studies}
            log(f"{target}: {min(counts)}-{max(counts)} run(s) a study, averaged as "
                f"ranks over {len(studies)} studies")
            _overwrite(frame, studies, [target], vals[:, None], log, blend=blend)
    finally:
        if owned is not None:
            shutil.rmtree(owned, ignore_errors=True)

    frame = frame.reset_index()[["StudyInstanceUID"] + TARGETS]
    frame.to_csv(out, index=False)
    log(f"wrote {out} — {len(frame)} studies")
    return out


def has_pixels(groups, n: int) -> np.ndarray:
    """Which of `n` studies had *any* pixel in *any* of an expert's regions.

    Judged across all regions rather than each, because an expert reads them together:
    a study the sagittal spec could not cut but the coronal one could has pixels, and
    the model saw them. Only a study masked everywhere saw nothing.

    The mask is `(studies, series, slots)`, so the reshape flattens everything but the
    study axis — it must not assume how many series or slots a spec has.
    """

    seen = np.zeros(n, bool)
    for _, mask in groups:
        seen |= np.asarray(mask).reshape(n, -1).any(axis=1)
    return seen


def _landmarks(landmark_runs, wanted, headers, device, workers, log) -> pd.DataFrame:
    """Every point the experts need, from whichever run provides it.

    A run is skipped when nothing depends on it, so shipping four landmark models and
    using three costs nothing. A point nobody provides is an error here rather than an
    empty region later, where it would read as "this study has no pixels".
    """

    tables, provided = [], set()
    for run in landmark_runs:
        _, config = load_landmarks(run, "cpu")
        if not set(config.points) & wanted:
            log(f"  {run.name}: provides {config.points}, nothing needs it — skipped")
            continue
        table, failed = predict_landmarks(run, headers, device=device,
                                          workers=workers, log=lambda m: log(f"  {m}"))
        tables.append(table)
        provided |= set(config.points)
        log(f"  {run.name}: {len(table)} rows, {len(failed)} studies failed")

    missing = wanted - provided
    if missing:
        raise ValueError(f"no landmark run provides {sorted(missing)}, which the "
                         f"regions hang off")
    return pd.concat(tables, ignore_index=True)


def _overwrite(frame: pd.DataFrame, studies, targets, scores: np.ndarray, log,
               blend: float | Mapping[str, float] | None = None) -> None:
    """Put an expert's ordering in, on the scale the column already uses.

    The expert's scores cannot simply be written in. `write_submission` emits per-column
    **percentile ranks**, because the metric is an AUC and ranks are what make two
    members blendable at all; an expert emits raw probabilities. A column holding ranks
    for the studies the expert could not read and probabilities for the rest is scored
    as one ordering, and the two halves do not belong to the same one — a wide rank of
    0.95 and an expert probability of 0.30 say nothing about each other.

    So the expert's values are quantile-mapped onto the ones already in the column for
    the same studies: lowest expert score takes the lowest value of that set, and so on.
    The ordering inside the scored group becomes exactly the expert's, which is the whole
    point, while the group keeps the place in the column it already had. When every study
    is scored — the usual case — this is a monotone remap of the whole column and changes
    no AUC at all. It only does work in the case that would otherwise be wrong.
    """

    index = pd.Index(studies)
    inside = index.isin(frame.index)
    rows = index[inside]
    if not len(rows):
        log("  no study of this expert is in the submission — column left alone")
        return
    used = {}
    for j, target in enumerate(targets):
        host = frame.loc[rows, target].to_numpy(float)
        mine = scores[inside, j]
        # A mapping gives each column its own weight. Measured out of fold, the optimum
        # ranges from 0.45 on MCL to 0.77 on Lateral Meniscus and tracks how far the
        # expert beats the host *alone* — so one constant is right only on average, and
        # the average happens to be 0.51, which is why a flat 0.5 has held up.
        #
        # A target the mapping does not name keeps the host's column untouched rather
        # than falling back to a default: naming eight columns and silently blending a
        # ninth is the kind of accident this signature exists to prevent.
        w = blend.get(target) if isinstance(blend, Mapping) else blend
        if isinstance(blend, Mapping) and w is None:
            log(f"  {target}: no weight given, column left to the host")
            continue
        used[target] = w
        if w is not None:
            # Both as ranks *within the scored studies*, which is what makes them
            # comparable at all: the expert emits probabilities and the column holds
            # percentile ranks over every study, scored or not.
            n = len(rows)
            mine = (w * (np.argsort(np.argsort(mine, kind="stable"),
                                    kind="stable") + 1) / n
                    + (1.0 - w) * (np.argsort(np.argsort(host, kind="stable"),
                                              kind="stable") + 1) / n)
        rank = np.argsort(np.argsort(mine, kind="stable"), kind="stable")
        frame.loc[rows, target] = np.sort(host)[rank]
    if not used:
        return
    how = ("outright" if all(w is None for w in used.values())
           else "blended at " + ", ".join(f"{t} {w:g}" for t, w in used.items())
           + " against the wide model")
    log(f"  {int(inside.sum())}/{len(frame)} studies scored by the expert ({how}); "
        f"the rest keep the wide model's answer")
