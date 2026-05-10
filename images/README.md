# Images

Static export target for the figures generated inside `notebooks/SendroffReidHW2.ipynb`. Plots in the notebook currently render inline only; the snippets below show the exact `savefig` calls to add at the end of each plotting cell so the figures land here for embedding in the top-level README.

## Catalog

### 1. `01_loss_vs_epoch_baseline.png`
- **Description.** Training and validation cross-entropy loss vs. epoch for the A2 baseline `SmallConvNet`.
- **Source.** Notebook cell 39 (problem A2 plot block).
- **Key finding to call out in the caption.** Train loss falls (3.82 → 3.48); val loss stays flat near 5.4. Classic underfit-on-the-train-side / mismatched-distribution-on-val-side signature when only 2 epochs and 20 train batches per epoch are used.
- **Save snippet to add at the end of the plotting block:**
  ```python
  plt.savefig("images/01_loss_vs_epoch_baseline.png", dpi=150, bbox_inches="tight")
  ```

### 2. `02_val_accuracy_vs_epoch_baseline.png`
- **Description.** Validation accuracy vs. epoch for the A2 baseline.
- **Source.** Notebook cell 39 (second `plt.figure` in the cell).
- **Key finding to call out in the caption.** Val accuracy ≈ 0.004 → 0.000 across two epochs; with 80 classes that is below random chance (≈0.013), a clear sign the model has not yet converged.
- **Save snippet:**
  ```python
  plt.savefig("images/02_val_accuracy_vs_epoch_baseline.png", dpi=150, bbox_inches="tight")
  ```

### 3. `03_training_percent_vs_loss.png`
- **Description.** Validation loss vs. percentage of training data used (5, 10, 20, 40, 60, 80, 100%).
- **Source.** Notebook cell 44 (sweep plot 1).
- **Key finding.** Loss is roughly flat across data fractions — at 2 epochs per trial the model has not converged enough for "more data" to translate into "lower loss." This is the experimental answer that motivated the B-track resolution study.
- **Save snippet:**
  ```python
  plt.savefig("images/03_training_percent_vs_loss.png", dpi=150, bbox_inches="tight")
  ```

### 4. `04_training_percent_vs_accuracy.png`
- **Description.** Validation accuracy vs. percentage of training data used.
- **Source.** Notebook cell 44 (sweep plot 2).
- **Key finding.** Val accuracy stays near 0 until 80–100%, where it nudges to 0.0037; consistent with the loss curve and confirms the underfit interpretation.
- **Save snippet:**
  ```python
  plt.savefig("images/04_training_percent_vs_accuracy.png", dpi=150, bbox_inches="tight")
  ```

### 5. `05_loss_vs_epoch_resolution128.png`
- **Description.** Train + val loss vs. epoch for the B2 6-epoch run at 128×128.
- **Source.** Notebook cell 63 (first plot).
- **Key finding.** Train loss declines monotonically (3.64 → 3.35); val loss is flat-to-rising. The architecture has begun to memorize the training subset but still cannot generalize at this batch / epoch budget.
- **Save snippet:**
  ```python
  plt.savefig("images/05_loss_vs_epoch_resolution128.png", dpi=150, bbox_inches="tight")
  ```

### 6. `06_val_accuracy_vs_epoch_resolution128.png`
- **Description.** Validation accuracy vs. epoch for the same B2 run.
- **Source.** Notebook cell 63 (second plot).
- **Save snippet:**
  ```python
  plt.savefig("images/06_val_accuracy_vs_epoch_resolution128.png", dpi=150, bbox_inches="tight")
  ```

### 7. `07_time_per_epoch.png`
- **Description.** Wall-clock seconds per epoch over the B2 run.
- **Source.** Notebook cell 63 (third plot).
- **Key finding.** ~200–260 s per epoch on CPU at 128×128, dominated by image decode + forward pass. This is the constraint that motivates running the final experiment on GPU.
- **Save snippet:**
  ```python
  plt.savefig("images/07_time_per_epoch.png", dpi=150, bbox_inches="tight")
  ```

### 8. `08_confusion_matrix.png` *(future work)*
- **Description.** Confusion matrix for the final test-set predictions.
- **Source.** To be added — the notebook does not currently emit one.
- **Suggested addition (after cell 69):**
  ```python
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  import numpy as np

  preds, labels = [], []
  model.eval()
  with torch.no_grad():
      for xb, yb in test_loader:
          out = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
          preds.extend(out); labels.extend(yb.numpy())
  cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
  fig, ax = plt.subplots(figsize=(14, 14))
  ConfusionMatrixDisplay(cm).plot(ax=ax, xticks_rotation=90, colorbar=False)
  plt.savefig("images/08_confusion_matrix.png", dpi=150, bbox_inches="tight")
  ```

## How to export

After the notebook has run end-to-end, append the relevant `plt.savefig(...)` line to each plotting cell (or paste the block below into a new cell after each plot):

```python
import matplotlib.pyplot as plt
plt.savefig("images/<name>.png", dpi=150, bbox_inches="tight")
```

Then commit only the `.png` files (each well under the 10 MB limit at 150 DPI).

## Image guidelines

- **Format:** PNG (raster) at 150 DPI is a good default; bump to 300 DPI only if a figure is going into a printed report.
- **Size:** keep individual files under 1 MB. The current set should land in the 100–400 KB range each.
- **Naming:** numeric prefix (`01_`, `02_`, …) preserves the order in directory listings and in this README.
- **Aspect ratio:** the default matplotlib `(6.4, 4.8)` works for line plots; use `(14, 14)` for the confusion matrix.

## Current status

> **TODO — none of the plots above have been exported yet.** Run the notebook end-to-end with the `savefig` snippets added, then drop the resulting PNGs into this directory and reference them from `../README.md` using:
>
> ```markdown
> ![Validation accuracy vs. training data %](images/04_training_percent_vs_accuracy.png)
> ```

## Embedding examples for the top-level README

Once the files exist, paste blocks like these into `README.md → ## Example Visualizations`:

```markdown
| Baseline learning curve                                  | Data-size sweep                                          |
|----------------------------------------------------------|----------------------------------------------------------|
| ![](images/01_loss_vs_epoch_baseline.png)                | ![](images/04_training_percent_vs_accuracy.png)          |
```
