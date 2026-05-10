# Data Sources

This document explains exactly what files this project expects, where they should live, and how to obtain them. Nothing under `100x100/` or `50x50/` is committed to the repository.

## Inventory

| File / directory     | Status            | Format                         | Expected location                       | Approx size |
|----------------------|-------------------|--------------------------------|-----------------------------------------|-------------|
| `truth.txt`          | Included          | TSV with header                | `./truth.txt` (also `100x100/truth.txt`) | ~2.3 MB     |
| `100x100/train/`     | Must download     | 31,991 JPEGs, 100×100 RGB      | `./100x100/train/<filename>.jpg`        | ~280 MB     |
| `100x100/validate/`  | Must download     | 6,000 JPEGs, 100×100 RGB       | `./100x100/validate/<filename>.jpg`     | ~52 MB      |
| `100x100/test/`      | Must download     | 4,882 JPEGs, 100×100 RGB       | `./100x100/test/<filename>.jpg`         | ~42 MB      |
| `50x50/train/`       | Must download     | 31,991 JPEGs, 50×50 RGB        | `./50x50/train/<filename>.jpg`          | ~125 MB     |
| `50x50/validate/`    | Must download     | 6,000 JPEGs, 50×50 RGB         | `./50x50/validate/<filename>.jpg`       | ~23 MB      |
| `50x50/test/`        | Must downlaod     | 4,882 JPEGs, 50×50 RGB         | `./50x50/test/<filename>.jpg`           | ~19 MB      |

## Source

The dataset was distributed as course material for **Harvard Extension School CSCI E-82 (Advanced Machine Learning, Fall 2025), Homework 2**. The expected upstream is the course's Google Drive bundle:

> `Drive/HW#2 Harvard ML Project/100x100/` (and sibling `50x50/`)

If you are enrolled in the course, the canonical bundle is reachable from the course's HW2 page on Canvas / the course Drive. <!-- TODO: replace with the public URL the instructor authorizes for redistribution, or a Kaggle/Zenodo mirror. -->

## Expected schema

Sample rows from `truth.txt`:

```
Filename	TrainValidateTest	Style	Genre	Artist
00000.jpg	train	landscape	Impressionism	Camille Pissarro
00001.jpg	train	landscape	Neo-Romanticism	Cookham
00005.jpg	train	abstract	Concretism	Richard Paul Lohse
00007.jpg	train	portrait	Post-Impressionism	Moise Kisling
```

- Tab-separated (`\t`).
- Header row is the literal `Filename TrainValidateTest Style Genre Artist`.
- 42,873 data rows; 31,991 / 6,000 / 4,882 across train / validate / test.
- 4 distinct `Style` values, 122 distinct `Genre` values.

## Download instructions

### Option A — course Drive (members only)

1. Mount Google Drive (in Colab) or download locally:
   ```
   /content/drive/MyDrive/HW#2 Harvard ML Project/100x100/
   ```
2. Copy the directory tree into this repo at `./100x100/`. Preserve the `train/`, `validate/`, `test/` subfolders.
3. Verify with the command in **Quick Setup** below.

### Option B — manual rebuild (advanced)

If you only have raw paintings and `truth.txt`, you can regenerate the resized variants:

```python
from pathlib import Path
from PIL import Image

SRC = Path("raw/")          # your full-resolution images
for size in (50, 100):
    out_root = Path(f"{size}x{size}")
    for split in ("train", "validate", "test"):
        (out_root / split).mkdir(parents=True, exist_ok=True)
    # then iterate truth.txt and write resized copies into the right split dir
```

## Quick Setup

After downloading, your project root should look like this:

```
cnn-art-genre-classification/
├── 100x100/
│   ├── train/      (31,991 *.jpg)
│   ├── validate/   ( 6,000 *.jpg)
│   ├── test/       ( 4,882 *.jpg)
│   └── truth.txt
├── 50x50/                 (optional, for CPU iteration)
│   └── …same shape…
├── truth.txt
└── notebooks/SendroffReidHW2.ipynb
```

Verify file counts with:

```bash
echo "100x100 train: $(ls 100x100/train | wc -l) (expected 31991)"
echo "100x100 valid: $(ls 100x100/validate | wc -l) (expected 6000)"
echo "100x100 test : $(ls 100x100/test | wc -l) (expected 4882)"
wc -l truth.txt   # expected: 42874 (header + 42873 rows)
```

## Alternative-data fallback

If you cannot obtain the full dataset, the notebook includes a `StratifiedShuffleSplit` fallback (cell 33) that will create a 70/15/15 stratified split from whatever rows are resolvable. You can also retarget any image-classification dataset with a `(filename, split, label)` schema by replacing `truth.txt` and pointing `DATA_ROOT` at the new image root — `FrameImageDataset` requires only the `filename` and `genre` columns and a flat `train/validate/test` directory layout.

## Privacy and licensing reminders

- The images depict publicly displayed works of art. No personal data is included.
- Verify your right to redistribute before mirroring this dataset publicly. If you publish a fork, prefer linking to the upstream rather than re-hosting.
- **Never commit files larger than 10 MB to GitHub.** The two image directories together exceed 500 MB and are excluded by `.gitignore`. If you accidentally stage them, run `git rm --cached -r 100x100 50x50` before committing.
