# Data

This project uses a labeled corpus of paintings provided by the Harvard Extension CSCI E-82 course (HW#2, 2025). Each image carries three categorical labels — **Style** (high-level subject), **Genre** (art-historical movement), and **Artist** — plus a `TrainValidateTest` column that fixes a deterministic split.

The raw image files are too large to ship in this repository; only the label index (`truth.txt`) is committed. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for download instructions.

## Datasets

### `truth.txt` — label index (committed)
- **Format:** tab-separated values with a header row.
- **Rows:** 42,873 labeled images.
- **Columns:**
  - `Filename` — image basename (e.g., `00000.jpg`)
  - `TrainValidateTest` — one of `train`, `validate`, `test`
  - `Style` — one of 4 high-level classes
  - `Genre` — one of 122 art-historical movements
  - `Artist` — free-form artist name
- **Splits:** train = 31,991 / validate = 6,000 / test = 4,882
- **Style distribution:**
  - portrait: 16,840
  - landscape: 15,005
  - abstract: 9,498
  - self-portrait: 1,530
- **Genre distribution (top 5 of 122):**
  - Realism: 5,454
  - Impressionism: 5,262
  - Romanticism: 4,012
  - Expressionism: 2,860
  - Post-Impressionism: 2,414
- **Long tail:** 7 genres have ≤3 examples (e.g., `International Gothic`, `Kinetic Art`, `Photorealism` each have 1). This is a deliberate property of the dataset — the assignment asks whether class size is the limiting factor for accuracy.

### `100x100/` — image resolution variant (not committed)
- **Format:** 8-bit JPEG, RGB, 100×100 pixels, square aspect.
- **Size on disk:** ~383 MB across train/validate/test directories.
- **Layout:** `100x100/{train,validate,test}/<filename>.jpg`
- A copy of `truth.txt` is included alongside the splits (same content as the top-level file).

### `50x50/` — image resolution variant (not committed)
- **Format:** 8-bit JPEG, RGB, 50×50 pixels.
- **Size on disk:** ~170 MB.
- **Layout:** `50x50/{train,validate,test}/<filename>.jpg`
- Use this when iterating on CPU; the 4× pixel reduction substantially cuts I/O and forward-pass time at the cost of fine-grained brushwork detail.

## Preprocessing applied in the notebook

1. **Filename resolution.** A first-pass walk of `DATA_ROOT` builds a `basename → relative_path` index, which is cached to `truth_resolved.csv` so subsequent runs skip the directory scan.
2. **Label cleanup.** Whitespace is stripped from `Genre`; rows with empty / `nan` / `none` genres are dropped.
3. **Split materialization.** Rows whose `trainvalidatetest` column matches `train`/`validate`/`test` are partitioned. If those splits are missing (e.g., when running against a partial Drive upload), a `StratifiedShuffleSplit` fallback creates a 70/15/15 stratified split, with rare-class fallback to a uniform random split.
4. **Image transforms.**
   - **Train:** `Resize((IMG_SIZE, IMG_SIZE)) → RandomHorizontalFlip → [optional ColorJitter] → ToTensor`
   - **Eval:** `Resize((IMG_SIZE, IMG_SIZE)) → ToTensor`
   - `IMG_SIZE` is swept across {64, 96, 128, 160, 224} depending on profile.

## Usage in code

```python
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("100x100")  # or "50x50" for the smaller variant
df = pd.read_csv(DATA_ROOT / "truth.txt", sep="\t")
df.columns = [c.strip().lower() for c in df.columns]

train_df = df[df["trainvalidatetest"] == "train"].reset_index(drop=True)
val_df   = df[df["trainvalidatetest"] == "validate"].reset_index(drop=True)
test_df  = df[df["trainvalidatetest"] == "test"].reset_index(drop=True)

# Each filename is a basename like "00000.jpg" — join with the split directory:
img_path = DATA_ROOT / "train" / train_df.loc[0, "filename"]
```

## Limitations and caveats

- **Class imbalance is severe.** The top 5 genres account for ~46% of labeled images; ~30% of genre classes have fewer than 50 examples. Any reported genre-classification accuracy must be paired with a per-class breakdown.
- **Random-chance accuracy** for 122-class genre prediction is ≈0.8%; for 4-class style prediction it is 25%.
- **Label noise.** Some artist/genre attributions in art-historical metadata are debated; this is unmodeled here.
- **Filenames are not stable identifiers across the 50x50 / 100x100 directories** — the same `00000.jpg` refers to the same painting in both, but a third-party copy of the dataset may number them differently. Always cross-check via `truth.txt`.

## Privacy and attribution

The images depict publicly displayed works of art assembled from open art-history collections for academic use. No personal data is involved. If you redistribute the dataset, retain attribution to the originating Harvard Extension CSCI E-82 course materials and the upstream art repositories they were aggregated from.

> **Do not commit `100x100/`, `50x50/`, or `truth_resolved.csv` back into git.** They are in `.gitignore`. Files larger than 10 MB will be rejected by GitHub's web upload path, and the dataset directories together exceed 500 MB.
