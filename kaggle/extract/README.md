# Extracting a subset

For getting a few dozen studies onto a laptop, without touching the file-download
endpoint.

## Why not just download the files

`kaggle competitions download -f <path>` fetches one named file; `-f` does not accept a
directory. A subset therefore costs **one API request per slice** — about 16,000 for 80
studies.

We tried it. It exhausted the account's download quota, after which *every* download
returned `429 Too Many Requests` — including the bulk transfer of the full corpus,
which is a different endpoint. Nothing could be fetched for hours.

The competition data is already mounted inside a Kaggle notebook. Archiving it there
and pulling the archive costs **one** request.

## Use

Set `N_STUDIES` in the notebook, then:

```bash
python -m scripts.kaggle_push --kernel-dir kaggle/extract
kaggle kernels status mathysgouverneur/rsna-knee-extract      # wait for complete
kaggle kernels output mathysgouverneur/rsna-knee-extract -p /tmp/extract
tar -xf /tmp/extract/train_subset.tar -C data/raw
```

The archive unpacks to `data/raw/train_series/<study>/<series>/*.dcm` — the layout
Kaggle itself mounts, so nothing downstream needs a different path.

## What the notebook guards against

**A truncated archive.** Kernel output is capped, so it asserts the total fits in 18 GB
and says what `N_STUDIES` would, rather than producing a short archive. A study missing
half its slices is not an error anywhere downstream: `read_slot` fills from the nearest
slice that decoded and `pick_slots` picks whichever series has the most. You would train
on quietly truncated data.

**A study cut in half.** It reads the archive back, counts slices per study, and fails
if any is missing — on Kaggle, before the download, rather than on the laptop after it.

## Not for the full corpus

`kaggle competitions download` without `-f` is already a single request for all 823 GB.
Use that on a machine with the disk for it.
