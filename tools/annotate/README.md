# Landmark annotation

A point per compartment horn, so a model can learn to find the meniscus and a ROI can be
cut around it. Three pieces, in the order they run.

```bash
# 1. build a bundle, on the machine that holds the DICOM
python -m tools.annotate.bundle --n 30 --out /data/mgr/rsna-knee/bundle-v1

# 2. serve it; VS Code forwards the port to your laptop's browser
cd /data/mgr/rsna-knee/bundle-v1 && python -m http.server 8000 --bind 127.0.0.1

# 3. annotate, press "export", then turn the download into millimetres
python -m tools.annotate.to_mm ~/Downloads/landmarks-*.json -o data/annotations/landmarks.csv
```

## Why it is shaped like this

**The bundle is self-sufficient.** 533 GB of DICOM becomes about 1 MB per study once
rendered at a resolution a human can click on. So the annotation session depends on
nothing — not the server being up, not the raw data being mounted, and not on having
access to it at all. Handing the directory to someone else is the whole distribution
story.

**A click is stored as a pixel on a named slice.** That is the only thing actually
observed. Patient millimetres are *derived* by `to_mm`, from the geometry the bundle
carries — so a mistake in the derivation costs a re-run, not six hundred re-clicks. Both
the raw and the derived columns end up in the CSV.

**Nothing is mirrored or reordered.** The bundle shows the acquisition as acquired. Which
end of the stack is lateral is *reported*, per study, so the annotator can see it — it is
not imposed by flipping pixels. A patient coordinate is independent of any convention we
might later change our mind about, and one of them is currently wrong: the sagittal stack
is not reversed for right knees, though `preprocessing.ipynb` says it should be. That bug
cannot reach an annotation stored in millimetres.

## What the tool refuses to guess

A study whose side could not be resolved is shown with a red banner and should be
**skipped**, not guessed — medial and lateral cannot be told apart on it, so an annotation
would be a coin toss recorded as fact. Same for a series whose geometry was unusable and
kept its arbitrary file order.

## Before annotating three hundred studies

Write the landmark definition down, with examples from `docs/atlas/`. Then both people
annotate the same twenty and run:

```bash
python -m tools.annotate.to_mm mine.json hers.json --agreement
```

That number is the floor. No model can be more accurate than the two people who defined
the landmark disagree with each other, and a 40 mm ROI tolerates about 12 mm. If the p90
is above that, the definition needs work — and it is much cheaper to learn that at forty
clicks than at six hundred.
