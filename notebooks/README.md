# Notebooks

This directory holds the Jupyter notebooks for the project.

## Files

### `SendroffReidHW2.ipynb` — main analysis
Implements **Option 2** of the CSCI E-82 HW#2 problem set: multi-class classification of paintings by **Genre** (122 art-historical movements in the full label index, 80–90 in the actual training subset), using a from-scratch CNN built layer-by-layer in PyTorch. The structure follows the assignment's problem ladder:

| Section          | Purpose                                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------------------|
| Setup            | Mount Drive (Colab), resolve image paths against `truth.txt`, build cached `truth_resolved.csv`               |
| Splits           | Materialize `train`/`validate`/`test` from the `TrainValidateTest` column, with stratified fallback            |
| Dataset & loaders| `FrameImageDataset` reads JPEGs lazily, applies resize + horizontal-flip augmentation, returns `(tensor, idx)` |
| Model            | `SmallConvNet`: 4 conv blocks (32→32→64→64→128→128→256 channels) → flatten → 512-d head → `num_classes` out    |
| **Problem A1**   | Experiment design: hold the architecture and hyperparameters fixed; vary % of training data                    |
| **Problem A2**   | Baseline CNN training run with loss/accuracy curves                                                            |
| Sweep helpers    | `stratified_percent` keeps every class represented at small fractions; `train_brief` runs short fits          |
| Sweep            | Train at {5, 10, 20, 40, 60, 80, 100}% of train data with pickled caching to `train_percent_sweep.pkl`        |
| **Problem A3**   | Hyperparameter optimization over `lr`, `weight_decay`, `dropout`, `img_size` (`TinyCNNLite` proxy for speed)  |
| **Problem A4**   | Test-set evaluation with partial-state-dict loading when class count drifts                                    |
| **Problem A5**   | Discussion of A's results                                                                                     |
| **Problem B1**   | Experiment design: hold dataset / hyperparameters; vary input resolution {64, 128, 224}                       |
| **Problem B2**   | Baseline 6-epoch run at 128×128                                                                               |
| **Problem B3**   | Grid over (resolution, lr, weight_decay, dropout) with capped batches per trial                               |
| **Problem B4**   | Test-set evaluation                                                                                           |
| **Problem B5**   | Discussion of B's results                                                                                     |
| C1 / C2          | Time spent (~11.5 hours), submission instructions                                                              |

### `HW2 CSCI E-82 2025.ipynb` — original assignment template
The unmodified course-issued problem set, kept here for reference. It contains the prompts for **all four** options (Novice / Experienced / Challenge / Challenge-at-scale); only Option 2 is solved in `SendroffReidHW2.ipynb`.

## How to run

### Local (CPU or GPU)

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows PowerShell
pip install -r requirements.txt
jupyter lab notebooks/SendroffReidHW2.ipynb
```

### Google Colab (recommended for full-data runs)

The notebook was authored on Colab with the dataset on Google Drive at `/content/drive/MyDrive/HW#2 Harvard ML Project/100x100`. To rerun:

1. Upload the `100x100/` folder to your Drive at the same path (or change `DATA_ROOT` in the setup cells).
2. Open the notebook in Colab.
3. Runtime → Change runtime type → **GPU** (a T4 is sufficient and dramatically reduces per-epoch time vs. CPU).
4. Run all cells. The first run rebuilds `truth_resolved.csv` and the sweep cache; subsequent runs reuse them.

## Key outputs (artifacts produced by running the notebook)

- `truth_resolved.csv` — basename → resolved relative path cache (regenerated, not committed).
- `train_percent_sweep.pkl` — list of `(percent, val_loss, val_acc)` tuples from the data-size sweep (cell 44).
- `checkpoints/*.pt` — per-trial PyTorch state dicts written by `save_checkpoint` (cell 37). Each carries `extra={img_size, lr, weight_decay, dropout, epochs}` for downstream comparison.
- Inline matplotlib plots: training/validation loss vs. epoch, validation accuracy vs. epoch, training-percent vs. validation accuracy, time-per-epoch vs. epoch.

## Notebook improvements (future work)

- [ ] Re-run end-to-end on the full 31,991-image train split (the captured outputs reflect the 1,801-row Drive subset that was loaded during the final pass — see the README's Results section).
- [ ] Promote epochs from 2–6 to 20–30 once on GPU; the captured curves show the model is still in its underfitting regime.
- [ ] Replace the `TinyCNNLite` hyperparameter sweep with `optuna` and a TPE sampler.
- [ ] Save final per-class confusion matrix to `images/` and link from the top-level README.
- [ ] Refactor cells 32, 33, 35 into a `src/dataset.py` module so the dataset/split/loader logic can be unit-tested.
- [ ] Strip the duplicate `import torch` cells (48 and 49 are identical) and remove the cell 26 environment-reset hack now that the notebook runs cleanly.
- [ ] Fix the cell 57 narrative: the printed `Test accuracy=0.0037` is 0.37%, not 96.3% — the discussion currently misreads accuracy as 1 − error.
