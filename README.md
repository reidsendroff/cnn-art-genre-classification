# CNN Art Genre Classification

> A from-scratch PyTorch CNN that classifies paintings by art-historical genre, with two designed experiments isolating the effect of training-set size and input resolution.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C?logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-%E2%89%A50.15-EE4C2C)
![pandas](https://img.shields.io/badge/pandas-%E2%89%A52.0-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.24-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.3-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%E2%89%A53.7-11557C)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Overview

This project is the Option 2 solution to **Homework 2 of Harvard Extension's CSCI E-82 (Advanced Machine Learning, Fall 2025)**. The assignment poses a single experimental question: when classifying paintings by art-historical genre across a 42,873-image corpus with **122 distinct genres**, is performance limited by the *number of classes* or by the *number of examples per class*? Rather than chase a single best accuracy number, the work designs two controlled experiments that vary one factor at a time so the contribution of each is legible in the curves.

The CNN is built **from primitive layers** in PyTorch — `Conv2d`, `ReLU`, `MaxPool2d`, `Dropout`, `Linear` — per the assignment's "no pre-built classifiers" rule. A 4-block convolutional stack feeds a 512-unit dense head; channel widths progress 32→32→64→64→128→128→256, and the input image size is parameterized so the same architecture runs at 64×64, 96×96, 128×128, 160×160, or 224×224. A lighter `TinyCNNLite` variant is used as a search proxy in the hyperparameter sweep to keep iteration tractable on CPU.

What makes the implementation distinctive is the engineering around training, not the architecture itself: per-class **stratified subsampling** to keep rare genres represented at small data fractions, **pickled sweep caches** so partial results survive notebook restarts, **per-trial checkpoint metadata** for after-the-fact comparison, and a **partial state-dict load** path that handles class-count drift between training and evaluation.

## What This Project Demonstrates

- Designing controlled experiments that isolate single variables (data quantity, then input resolution)
- Building CNNs layer-by-layer in PyTorch rather than using high-level model APIs
- Engineering a custom `Dataset` for tab-separated label indices with lazy JPEG decoding
- Stratified per-class sampling for severely long-tailed categorical labels
- Reproducible CPU-friendly hyperparameter sweeps with cached intermediates
- Honest reporting of underfit regimes and identification of the binding compute constraints
- Checkpointing with embedded metadata, including partial-load fallback when class counts shift

## Methods Used

| Method                                | Type                              | Use Case                                                                 |
|---------------------------------------|-----------------------------------|--------------------------------------------------------------------------|
| `SmallConvNet` (4-block CNN)          | Supervised image classifier        | Final classifier; trained at 64–224 px input                              |
| `TinyCNNLite` (3-block CNN)           | Lightweight search proxy           | Hyperparameter sweeps where per-trial wall time matters more than ceiling |
| Cross-entropy loss + Adam             | Optimization                       | 122-way (in practice 80–90-way) softmax classification                    |
| `RandomHorizontalFlip` + `ColorJitter`| Data augmentation                  | Reduce overfit on long-tail genres                                        |
| Stratified per-class subsampling      | Sampling                           | Keep all genres present at 5–80% data fractions                           |
| `StratifiedShuffleSplit` fallback     | Split materialization              | Recreate train/val/test when the original split column is unavailable     |
| Pickled `(percent, val_loss, val_acc)` cache | Iteration speedup           | Resume the data-fraction sweep after notebook restarts                    |
| Per-trial `state_dict + extra` checkpoints | Experiment tracking            | Compare runs by `(img_size, lr, wd, dropout)` after the fact              |
| Partial `state_dict` load + classifier-head fine-tune | Robustness            | Evaluate a checkpoint trained on 82 classes against a 90-class test set   |

## Datasets / Inputs

- **Source:** Harvard Extension CSCI E-82 (Fall 2025) HW#2 dataset.
- **Size:** 42,873 labeled images — 31,991 train / 6,000 validate / 4,882 test.
- **Resolutions provided:** 100×100 (~383 MB) and 50×50 (~170 MB).
- **Labels** (in `truth.txt`): `Filename`, `TrainValidateTest`, `Style`, `Genre`, `Artist`.
  - **Style** has 4 classes — portrait (16,840), landscape (15,005), abstract (9,498), self-portrait (1,530).
  - **Genre** has 122 classes — top of the distribution: Realism (5,454), Impressionism (5,262), Romanticism (4,012), Expressionism (2,860), Post-Impressionism (2,414). Tail: 7 genres with ≤3 examples (e.g., `International Gothic`, `Kinetic Art`, `Photorealism` each have 1).
- **Preprocessing applied:** filename resolution against `DATA_ROOT` (cached to `truth_resolved.csv`); whitespace strip + `nan`/`none` filtering on `Genre`; `Resize` + `ToTensor`; `RandomHorizontalFlip` (and optionally `ColorJitter(0.2, 0.2, 0.2, 0.1)`) at train time only.
- See **[`data/README.md`](data/README.md)** and **[`data/DATA_SOURCES.md`](data/DATA_SOURCES.md)** for full schema and download instructions. The image directories are not committed.

## Key Technical Steps

```
truth.txt ──► resolve filenames ──► train_df / val_df / test_df
                                          │
                                          ▼
                              FrameImageDataset (lazy PIL decode + transforms)
                                          │
                                          ▼
                              DataLoader (batches of 64)
                                          │
                                          ▼
                              SmallConvNet  ──►  cross-entropy loss  ──►  Adam
                                          │
                                          ▼
                              history{train_loss, val_loss, val_acc}
                                          │
                                          ▼
                              save_checkpoint(state_dict + extra metadata)
```

### 1. Label index and split materialization
- `load_df()` (notebook cell 32) reads `truth.txt` (TSV), lowercases column names, and builds a `basename → relative_path` index so filenames in `truth.txt` are decoupled from on-disk layout. The result is cached to `truth_resolved.csv`.
- Splits are pulled from the `TrainValidateTest` column. If those splits are empty (e.g., the Drive copy was partial), a `StratifiedShuffleSplit(n_splits=1, test_size=0.30)` followed by a 50/50 split of the remainder produces a stratified 70/15/15 fallback. Genres with fewer than 2 examples short-circuit to a uniform random split to avoid the stratifier's two-sample minimum.

### 2. `FrameImageDataset` — custom PyTorch `Dataset`
- Accepts a DataFrame, a root path, a transform, and a `label_col`.
- Picks `resolved` if present, otherwise joins `DATA_ROOT / filename`.
- Builds a stable `cls2i = {class_name: index}` map by sorting unique labels at construction time so the same model can later be reloaded against the same label space.
- `__getitem__` opens with `Image.open(...).convert("RGB")` and returns `(tensor, class_idx)`.

### 3. `SmallConvNet` — the classifier
The architecture, parameterized by `img_size` and `drop`:

```
Conv(3→32, 3×3, pad=1) → ReLU → Conv(32→32, 3×3) → ReLU → MaxPool(2)
Conv(32→64, 3×3)       → ReLU → Conv(64→64, 3×3) → ReLU → MaxPool(2)
Conv(64→128, 3×3)      → ReLU → Conv(128→128, 3×3) → ReLU → MaxPool(2)
Conv(128→256, 3×3)     → ReLU → MaxPool(2)
Flatten → Linear(256·(img_size/16)², 512) → ReLU → Dropout(drop) → Linear(512, num_classes)
```

Loss: `nn.CrossEntropyLoss()`. Optimizer: `optim.Adam(lr=LR, weight_decay=WEIGHT_DECAY)`.

### 4. Training loop (cell 39)
- Standard `model.train() → forward → backward → step` with `optimizer.zero_grad(set_to_none=True)`.
- Optional `MAX_TRAIN_BATCHES` / `MAX_VAL_BATCHES` caps for fast iteration on CPU.
- Per-epoch metrics appended to `history = {"train_loss": [...], "val_loss": [...], "val_acc": [...]}`.

### 5. Experiment A — training-fraction sweep (cells 43–44)
- `stratified_percent(frame, percent)` ensures **every** class contributes at least one sample at every fraction.
- Trains `train_brief(...)` at each of `[5, 10, 20, 40, 60, 80, 100]%`.
- Each result `(percent, val_loss, val_acc)` is appended to a list and pickled to `train_percent_sweep.pkl` so the sweep is resumable across notebook restarts.

### 6. Experiment B — resolution sweep (cells 67, 63)
- Search space: `RES ∈ {64, 128, 224}` × `LR ∈ {5e-4, 1e-3}` × `WD ∈ {0.0, 1e-4}` × `DROP ∈ {0.0, 0.3}`.
- Per-trial speed cap: `MAX_TRAIN_BATCHES = 80`, `MAX_VAL_BATCHES = 40`, `EPOCHS_SMALL = 3`.
- Each trial records `{val_acc, val_loss, train_loss, time_per_epoch_s, wall_s}` plus the configuration.

### 7. Test-time evaluation (cell 55)
- The best validation checkpoint (by `max history["val_acc"]`) is reloaded.
- If `head.4.weight` rows ≠ current `num_classes`, the classifier head is dropped from the state dict and the rest is loaded with `strict=False`. The feature extractor is then frozen and the new head is fine-tuned for 1–2 epochs before evaluating against the test loader.

## Results and Interpretation

The numbers below are the actual values printed by the captured notebook outputs. They reflect a CPU run against the **partial Drive-uploaded subset** (1,801 rows resolvable, 90 unique genres) with batch caps and 2–6 epochs per trial.

### Baseline `SmallConvNet` at 128×128 (cell 39 output)

| Epoch | Train loss | Val loss | Val acc  |
|-------|-----------:|---------:|---------:|
| 1     | 3.8179     | 5.4052   | 0.0037   |
| 2     | 3.4842     | 5.3667   | 0.0000   |

Training loss falls (3.82 → 3.48); validation loss is flat near 5.4. With 80 active classes the random-chance accuracy is ≈0.0125, so the baseline at this budget is below chance — clear underfit.

### Training-fraction sweep (cell 44 output)

| Train %  | Val loss | Val acc   |
|---------:|---------:|----------:|
| 5        | 4.6176   | 0.0000    |
| 10       | 4.5889   | 0.0000    |
| 20       | 4.5909   | 0.0000    |
| 40       | 5.4571   | 0.0000    |
| 60       | 5.4855   | 0.0000    |
| 80       | 5.7314   | 0.0037    |
| 100      | 5.6677   | 0.0037    |

At 2 epochs per trial, "more data" does not yet translate into "lower loss" — the model has not converged. The sweep correctly establishes the experimental harness; the conclusion is that *epochs*, not *data fraction*, is the binding constraint at this compute budget.

### Hyperparameter sweep (cell 48 output)

| Checkpoint                                                    | Best val acc | IMG | LR     | WD     | Dropout |
|---------------------------------------------------------------|-------------:|----:|-------:|-------:|--------:|
| `tuned_IMG128_E2_LR0.0005_WD0.0001_DO0.5.pt`                  | 0.0314       | 128 | 5e-4   | 1e-4   | 0.5     |
| `tuned_IMG128_E2_LR0.002_WD0.0_DO0.3.pt`                      | 0.0314       | 128 | 2e-3   | 0.0    | 0.3     |
| `tuned_IMG128_E2_LR0.001_WD0.0001_DO0.3.pt`                   | 0.0111       | 128 | 1e-3   | 1e-4   | 0.3     |
| `baseline_IMG128_E2_LR0.001_WD0.0001.pt`                      | 0.0037       | 128 | 1e-3   | 1e-4   | —       |
| `baseline_IMG96_E1_LR0.001_WD0.0001.pt`                       | 0.0000       |  96 | 1e-3   | 1e-4   | —       |

Best captured val accuracy is **3.14%** — ~2.5× random chance — at `img_size=128, lr=5e-4, dropout=0.5`. Higher resolution did not help meaningfully at this epoch budget; the gain came from the regularization combination.

### Resolution sweep at 128×128, 6 epochs (cell 63 output)

| Epoch | Train loss | Val loss | Val acc | Time/epoch |
|------:|-----------:|---------:|--------:|-----------:|
| 1     | 3.6404     | 5.4785   | 0.0111  | 202.5 s    |
| 2     | 3.4067     | 5.5576   | 0.0000  | 198.6 s    |
| 3     | 3.3856     | 5.5812   | 0.0111  | 215.6 s    |
| 4     | 3.3886     | 5.5678   | 0.0111  | 252.4 s    |
| 5     | 3.3630     | 5.7010   | 0.0111  | 258.7 s    |
| 6     | 3.3513     | 5.4150   | 0.0037  | 248.4 s    |

Train loss continues to decline; val loss does not. Per-epoch wall time is 200–260 s on CPU — the constraint that motivates GPU re-runs.

### Test-set evaluation (cell 55 output)

```
Evaluating: tuned_IMG128_E2_LR0.0005_WD0.0001_DO0.5.pt
[load] strict=False | missing: ['head.4.weight', 'head.4.bias'] | unexpected: []
[adapt] class count changed (ckpt: 82 → now: 90). Fine-tuning classifier briefly…
[B4] Test loss=5.5292 | Test accuracy=0.0037
```

After the partial-load + brief head fine-tune, the test accuracy is **0.37%** — below random chance for 90 classes (≈1.1%), reflecting that the head was re-initialized and only briefly adapted before evaluation. The captured run is a methodology exercise; the planned full-data, GPU, ~25-epoch re-run is the one that should produce the headline number.

> Calibrating expectations: at random chance, val accuracy ≈ 1/80 ≈ 1.25%. The 3.14% best is a 2.5× lift but still far from a useful classifier. The Future Improvements list below is the path to closing that gap.

## Example Visualizations

Plots are produced inline by the notebook. The catalog of figures to export to `images/` (with the exact `plt.savefig` snippets) lives in **[`images/README.md`](images/README.md)**. Once exported, embed them here:

| Figure                                                          | Source           |
|-----------------------------------------------------------------|------------------|
| Train + val loss vs. epoch (baseline)                           | cell 39          |
| Validation accuracy vs. epoch (baseline)                        | cell 39          |
| Validation loss vs. % training data                             | cell 44          |
| Validation accuracy vs. % training data                         | cell 44          |
| Train + val loss vs. epoch at 128×128 (B2 6-epoch run)          | cell 63          |
| Validation accuracy vs. epoch at 128×128                        | cell 63          |
| Wall-clock seconds per epoch (CPU, 128×128)                     | cell 63          |
| **TODO** confusion matrix on the test set                       | (not yet emitted)|

## Repository Structure

```
cnn-art-genre-classification/
├── README.md                         (this file)
├── PROJECT_SUMMARY.md                (portfolio-friendly elevator pitch)
├── requirements.txt                  (pinned dependency floors)
├── .gitignore                        (Python + Jupyter + dataset excludes)
├── truth.txt                         (label index, 42,874 lines incl. header)
├── data/
│   ├── README.md                     (schema, splits, preprocessing details)
│   └── DATA_SOURCES.md               (where to download images, expected layout)
├── notebooks/
│   ├── README.md                     (per-section walkthrough of the notebook)
│   ├── SendroffReidHW2.ipynb         (main analysis — Option 2 solution)
│   └── HW2 CSCI E-82 2025.ipynb      (original assignment template, all options)
├── images/
│   └── README.md                     (catalog + savefig snippets; PNGs to be exported)
└── reports/
    └── README.md                     (PDF / HTML export instructions)
```

The `100x100/` and `50x50/` image directories are present locally during development but are excluded from version control by `.gitignore`.

## How to Run

### Prerequisites
- Python 3.10 or newer
- (Recommended) An NVIDIA GPU with CUDA, or Apple Silicon with MPS — CPU works but is the bottleneck
- ~600 MB of free disk for the image datasets
- Optional: a working LaTeX or headless Chrome install if you want to regenerate the PDF report

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-github-username>/cnn-art-genre-classification.git
cd cnn-art-genre-classification

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (see data/DATA_SOURCES.md) and unpack to:
#    100x100/{train,validate,test}/ + 100x100/truth.txt
#    50x50/{train,validate,test}/   + 50x50/truth.txt   (optional, smaller)

# 5. Verify the file counts
ls 100x100/train | wc -l    # expected: 31991
ls 100x100/validate | wc -l # expected: 6000
ls 100x100/test | wc -l     # expected: 4882

# 6. Launch the notebook
jupyter lab notebooks/SendroffReidHW2.ipynb
```

For Colab + Drive runs (the original authoring environment), see `notebooks/README.md`.

## Skills Demonstrated

**Mathematical / statistical**
- Cross-entropy / softmax classification on a long-tailed multi-class label space
- Stratified sampling for severely imbalanced labels
- Hyperparameter sweeps as low-budget grid search; reading curves to identify the binding constraint
- Diagnosing underfit vs. overfit from the gap between train loss and val loss

**Programming and tools**
- PyTorch `nn.Module` subclassing, custom `Dataset`, `DataLoader`, `state_dict` serialization, partial `state_dict` loading
- torchvision transforms pipeline (`Resize`, `RandomHorizontalFlip`, `ColorJitter`, `ToTensor`)
- pandas (TSV ingest, label cleanup, stratified `groupby` sampling)
- numpy
- scikit-learn `StratifiedShuffleSplit`
- PIL for lazy JPEG decoding
- matplotlib for training/sweep curves
- Jupyter / Google Colab + Drive integration
- pickle-based intermediate caching

**Workflow**
- Reproducible iteration via cached intermediates (`truth_resolved.csv`, `train_percent_sweep.pkl`, per-trial `*.pt`)
- Profile-based speed/quality presets (`speed_probe`, `debug_cpu`, `cpu_balanced`, `final`)
- Designed experiments that vary one factor at a time
- Honest reporting of underfit regimes — calling out what the captured numbers do and don't say

## Project Context

This is the **Option 2** solution to **Harvard Extension School CSCI E-82, "Advanced Machine Learning," Fall 2025, Homework 2**. The course explicitly asks students to *build their own networks* (i.e., compose layers, no `torchvision.models`) and to *think through experiments* that produce evidence rather than just chase benchmark numbers. Option 2 is "for those with CNN experience" — multi-class genre prediction with the long-tailed label distribution surfaced explicitly so the student has to make defensible choices about merging or dropping rare classes.

## Why This Matters

Art-historical genre classification is a small, friendly proxy for a much broader problem: **fine-grained classification with long-tailed labels.** The same shape of problem shows up in product taxonomies on e-commerce sites, ICD-10 medical coding, scientific paper subject classification, biological species identification, and content moderation taxonomies. The two questions this project asks — *how much data per class, and how much input fidelity, do you need before adding more stops paying off?* — are the questions a practitioner will face on every one of those problems. Designing the experiment matters as much as the model.

## Future Improvements

- [ ] Re-run the full pipeline on GPU (Colab T4 or better) with the full 31,991-image train split — the captured outputs reflect a 1,801-row Drive subset.
- [ ] Bump epochs from 2–6 per trial to 20–30 once on GPU; the curves indicate the model is still in the underfit regime.
- [ ] Add a per-class confusion matrix on the test set (snippet in `images/README.md`) to surface which genres are confusable vs. distinguishable.
- [ ] Replace the manual grid sweep with `optuna` + a TPE sampler for a more efficient hyperparameter search.
- [ ] Add transfer-learning baseline (frozen ImageNet ResNet-18 features → linear head) as a control for the from-scratch CNN.
- [ ] Explore class-merging strategies (e.g., the assignment's "Cloisonnism with 46 examples — enough?" question) — group rarest 30 genres into an `other` bucket and quantify the accuracy cost.
- [ ] Refactor cells 32, 33, 35 into a `src/dataset.py` module; add unit tests for the filename resolver and the stratified sampler.
- [ ] Strip duplicate cells (48 ≈ 49) and the cell 26 environment-reset hack now that the notebook is stable.
- [ ] Fix the cell 57 narrative misreading (`Test accuracy=0.0037` is 0.37%, not 96.3%).
- [ ] Export plots to `images/` and embed them above using the catalog in `images/README.md`.

## Author

**Reid Sendroff**
Harvard Extension School — CSCI E-82 (Advanced Machine Learning), Fall 2025

---

*Built layer-by-layer in PyTorch, designed experiment-first, and honest about what the captured runs do and don't say.*
