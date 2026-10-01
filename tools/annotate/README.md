# Landmark annotation

One point per study — the centre of the lateral meniscus — so a model can learn to find
it and a ROI can be cut around it. [What the point means](LANDMARK.md) is the definition;
read it before the first click. Three pieces, in the order they run.

```bash
# 1. choose what goes in — a random draw reproduces whatever bias the last one had
python -m tools.annotate.select --scan /data/mgr/rsna-knee/study_scan.csv \
    --exclude /data/mgr/rsna-knee/bundle-v2/studies.json --total 300 -o studies-v3.txt

# 2. render it, on the machine that holds the DICOM
python -m tools.annotate.bundle --studies studies-v3.txt --out /data/mgr/rsna-knee/bundle-v3

# 3. serve it; VS Code forwards the port to your laptop's browser
cd /data/mgr/rsna-knee/bundle-v3 && python -m http.server 8000 --bind 127.0.0.1

# 4. annotate, press "export", then turn the download into millimetres
python -m tools.annotate.to_mm ~/Downloads/landmarks-*.json -o data/manual_annotations/landmarks.csv
```

`--n 30` still draws at random when no list is given, which is how the first bundle was
built. That turned out to be the mistake worth naming: see below.

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
not imposed by flipping pixels. That is what makes the annotation outlive our conventions:
the training pipeline reverses right-knee sagittal stacks or not depending on
`rules.sagittal_flip`, and a coordinate in patient millimetres is unaffected either way.

## What the tool refuses to guess

A badge derived from the median patient x is a guess — right about 91 % of the time —
and is drawn differently from one read off the DICOM tag, so the weaker answer cannot
borrow the stronger one's authority. Where there is no badge at all, the annotator orients
on the fibular head, which exists only on the lateral side.

Nothing extra is entered. **The click already carries the answer**: the lateral meniscus
sits about 25 mm from the lateral edge of a knee roughly 80 mm wide, so a point on it lands
distinctly nearer one end of the stack, and `to_mm` reads the lateral end back off it —
161/161 against the tag on the first bundle, with the most central click still 31 % of the
half-stack from the middle. Asking the annotator to restate it would add a field and no
information.

Where a tag exists the click restates it, so the two can disagree, and `to_mm` prints the
disagreements. A point on the wrong compartment is the one error this dataset cannot
survive, and it is the only thing that would report it.

That is a change from the first bundle, which was built `--tagged-only` and told the
annotator to skip these. The filter was sound in itself — the geometric fallback disagrees
with the tag on 9.4 % of studies where both exist — but it selected a *manufacturer*: the
`Laterality` tag is written by whole vendors, so the bundle came out 78.8 % Siemens against
45 % of the corpus, with GE down to three studies out of 170. A landmark model trained on
it would have been blind to 22 % of the corpus with nothing reporting it.

A series whose geometry was unusable keeps its arbitrary file order and should still be
skipped: "which end is lateral" means nothing on a stack that is not in anatomical order.

## Deep series

A 3D acquisition carries 320 slices at 0.6 mm where a 2D one carries 30 at 3.4 mm.
Rendering all of them would cost 14 MB per study and make the stack unscrollable, and buy
nothing — the landmark is one point, and its depth precision is set by how well a human can
see the horns. Stacks over `MAX_SLICES` are therefore subsampled **in depth only**. Every
frame stays a real acquired slice keyed by its own `SOPInstanceUID`, so a click still
resolves exactly; `native_n` and `native_spacing_mm` record what was thinned.

## Before annotating a hundred and seventy studies

The definition is written: [LANDMARK.md](LANDMARK.md), and the same text is in the tool
under <kbd>H</kbd> so it is one key away at the moment of clicking rather than something
read once and drifted from. Both people annotate the same twenty first, then run:

```bash
python -m tools.annotate.to_mm mine.json hers.json --agreement
```

That number is the floor. No model can be more accurate than the two people who defined
the landmark disagree with each other, and a 40 mm ROI tolerates about 12 mm. If the p90
is above that, the definition needs work — and it is much cheaper to learn that at forty
clicks than at six hundred.
