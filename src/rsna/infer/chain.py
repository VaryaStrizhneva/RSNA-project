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
could not be predicted, or whose series the region needs is missing, keeps the wide
model's answer instead of a prior — and keeps a row, which the submission format
requires whatever happened upstream.

**One region at a time.** Cutting all seven at once is 10.3 GB for 1300 studies; cutting
one, scoring it and dropping it is 1.6 GB at the peak. The scored notebook has 20 GB of
writable disk and other things to put in it.

**Only files this function wrote are deleted.** Pointed at the training cache directory
it will reuse what is already cut rather than redo it, and will not remove it. A rehearsal
that deletes the caches it was asked to verify against is a bad afternoon.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import TARGETS, Config
from ..dicom import annotate, walk
from ..roi import cache
from .experts import load_experts, score_experts
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
    owners: dict = {}
    for run, (_, config, _) in loaded.items():
        for target in config.targets:
            if target in owners:
                raise ValueError(
                    f"{run} and {owners[target]} both claim {target!r}; the submission "
                    f"would take whichever ran last, which is not a decision")
            owners[target] = run
    return loaded, wanted


def run_expert_submission(expert_runs, landmark_runs, package, data_root,
                          split: str = "test_series", encoder=None,
                          out="submission.csv", scratch=None, device: str = "cpu",
                          workers: int = 8, keep_cache: bool = False,
                          blend: float | None = None, log=print) -> Path:
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
    """

    if data_root is None:
        raise ValueError("data_root is None — the caller did not resolve the competition "
                         "root, and pathlib's error for that is three frames deep")
    data_root, out = Path(data_root), Path(out)
    expert_runs = [Path(r) for r in expert_runs]
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
    made: list[Path] = []

    try:
        points = _landmarks(landmark_runs, wanted, headers, device, workers, log)
        table = room / "landmarks.csv"
        points.to_csv(table, index=False)

        for run in expert_runs:
            models, config, specs = loaded[str(run)]
            log(f"{run.name}: {', '.join(config.targets)}")
            groups, names = [], None
            for spec in specs:
                tag = cache.cache_tag(spec)
                if not (room / f"{tag}.json").exists():
                    cache.build(table, headers, room, spec, Config(),
                                workers=workers, log=lambda m: log(f"    {m}"))
                    made += [room / f"{tag}.npy", room / f"{tag}_mask.npy",
                             room / f"{tag}.json"]
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
            _overwrite(frame, names, config.targets, scores, log, blend=blend)

            if not keep_cache:
                for path in list(made):
                    if path.exists():
                        path.unlink()
                    made.remove(path)
    finally:
        if owned is not None:
            shutil.rmtree(owned, ignore_errors=True)

    frame = frame.reset_index()[["StudyInstanceUID"] + TARGETS]
    frame.to_csv(out, index=False)
    log(f"wrote {out} — {len(frame)} studies")
    return out


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
               blend: float | None = None) -> None:
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
    for j, target in enumerate(targets):
        host = frame.loc[rows, target].to_numpy(float)
        mine = scores[inside, j]
        if blend is not None:
            # Both as ranks *within the scored studies*, which is what makes them
            # comparable at all: the expert emits probabilities and the column holds
            # percentile ranks over every study, scored or not.
            n = len(rows)
            mine = (blend * (np.argsort(np.argsort(mine, kind="stable"),
                                        kind="stable") + 1) / n
                    + (1.0 - blend) * (np.argsort(np.argsort(host, kind="stable"),
                                                  kind="stable") + 1) / n)
        rank = np.argsort(np.argsort(mine, kind="stable"), kind="stable")
        frame.loc[rows, target] = np.sort(host)[rank]
    how = "outright" if blend is None else f"blended at {blend:g} against the wide model"
    log(f"  {int(inside.sum())}/{len(frame)} studies scored by the expert ({how}); "
        f"the rest keep the wide model's answer")
