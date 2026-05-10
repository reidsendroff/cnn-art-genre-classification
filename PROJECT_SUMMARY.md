# Project Summary — CNN Art Genre Classification

## Concise Summary

A from-scratch convolutional neural network in PyTorch that classifies paintings by art-historical **genre** (122 classes in the full label index, 80–90 in the working subset) on a 42,873-image dataset of styles, landscapes, portraits, and abstracts. Two designed experiments isolate the effect of (A) **how much** training data is needed and (B) **how detailed** the input image needs to be — holding architecture and hyperparameters fixed in each. The architecture is a 4-block CNN with ReLU + MaxPool + dropout, sized to be runnable on CPU and scalable to GPU. The headline finding from the captured runs is that under tight CPU/epoch budgets the model lives in its underfitting regime — the experiments establish the methodology and identify the binding constraints (epochs, batch coverage, GPU access) before scaling up.

## Resume Bullets

- **Designed and trained a multi-class CNN** in PyTorch from primitive layers (no high-level model APIs) for 80+ class art-genre classification on a 42K-image painting corpus, including a custom `Dataset` with stratified sampling for severely imbalanced labels.
- **Engineered two controlled experiments** isolating training-set size and input resolution as independent variables, with cached pickle/checkpoint pipelines that cut iteration time by avoiding redundant re-training across notebook reloads.
- **Built a CPU-first hyperparameter sweep harness** with profile-based speed presets (`speed_probe` / `debug_cpu` / `cpu_balanced` / `final`) and per-trial checkpoint metadata, enabling reproducible comparison of `lr`, `weight_decay`, `dropout`, and `img_size` settings.

## Technical Explanation

The pipeline ingests `truth.txt` (TSV with `filename`, `trainvalidatetest`, `style`, `genre`, `artist`), resolves filenames against `DATA_ROOT` once and caches the result to `truth_resolved.csv` to skip directory walks on subsequent runs. A `FrameImageDataset` lazily decodes JPEGs with `PIL`, applies `Resize` + `RandomHorizontalFlip` + (optional) `ColorJitter` + `ToTensor`, and emits class indices via a sorted-genre `cls2i` map. The model — `SmallConvNet` — is four conv blocks (`32→32→64→64→128→128→256` channels) with `ReLU` activations and `MaxPool2d` between blocks, flattened into a 512-unit dense head with dropout. Training uses cross-entropy loss and the Adam optimizer; the training loop tracks per-epoch train loss, validation loss, and validation accuracy in a `history` dict for plotting. A second, lighter `TinyCNNLite` is used as a search proxy in the hyperparameter sweep to keep CPU iteration tractable. Checkpoints (state-dict + history + `extra` metadata) are written per trial; a partial-load path handles class-count drift between the cached subset and the live data so a checkpoint trained on 82 classes can be evaluated on a 90-class test set after a brief classifier-head fine-tune.

## Interview Version

In this project, I built a CNN from scratch in PyTorch to classify paintings by art-historical genre — about 80 classes in practice, drawn from a label index of 122. I picked Option 2 of a Harvard Extension graduate-level homework, which asks the question *how much training data, and at what resolution, do you actually need?* I designed two experiments to answer that. In the first, I held the architecture and hyperparameters fixed and swept the percentage of training data from 5% to 100% with stratified per-class sampling so even the rarest genres stayed represented. In the second, I held the data fixed and swept input resolution across 64, 128, and 224 pixels. I built a small custom `Dataset` class that lazily loads images, a `SmallConvNet` with four conv blocks plus a dropout head, and a sweep harness with pickle-based caching and per-trial checkpoint metadata so I could reload partial results across notebook restarts instead of re-training from zero. The captured runs were CPU-bound and stayed in the underfitting regime — the most interesting outcome wasn't a high accuracy but realizing that the experiments cleanly identified the binding constraints (epochs, GPU access, batch coverage), which is the right kind of negative result to act on. Next pass is to rerun on GPU with an order-of-magnitude more epochs and add a per-class confusion matrix.

## Why This Project Stands Out

- **Built layer-by-layer, not via `torchvision.models`.** Per the assignment's "no pre-built classifiers" rule, every Conv/ReLU/MaxPool/Dropout/Linear is wired explicitly.
- **Honest experimental design.** The two experiments isolate single variables (data quantity, then input resolution). The reported numbers reflect what actually ran rather than a tuned best-of-many.
- **Deliberate engineering for slow hardware.** Speed profiles, pickled sweep cache, partial-state-dict loading, and per-class stratified subsampling were all added because the work happened on CPU, where every minute of training has to count.
- **Diagnoses its own ceiling.** The notebook surfaces the underfitting regime instead of glossing over it, and the discussion in `SendroffReidHW2.ipynb` (problems A5 and B5) calls out exactly what to change to push performance up.

## Key Skills Demonstrated

**Mathematical / statistical**
- Cross-entropy loss, softmax classification with severe class imbalance
- Stratified sampling for long-tailed categorical labels
- Hyperparameter sweeps as low-budget grid search

**Programming and tools**
- PyTorch (custom `nn.Module`, `Dataset`, `DataLoader`, `state_dict` partial loads, checkpointing)
- torchvision transforms pipeline
- pandas / numpy for label management
- scikit-learn `StratifiedShuffleSplit` for the fallback split path
- matplotlib for training curves
- PIL for lazy image I/O
- Google Colab + Drive integration

**Workflow**
- Reproducibility via cached intermediates (`truth_resolved.csv`, `train_percent_sweep.pkl`, per-trial `*.pt`)
- Profile-based speed/quality presets for CPU-vs-GPU iteration
- Designed experiments that vary one factor at a time
- Honest reporting of underfitting and clear "what to do next" recommendations

## Project Outcomes

- **42,873 images** indexed across **122 distinct genres** with a 31,991 / 6,000 / 4,882 train/validate/test split.
- **Two designed experiments** (training-fraction sweep, resolution sweep) executed end-to-end with cached, resumable runs.
- **Best captured validation accuracy:** **3.14%** at `img_size=128`, `lr=5e-4`, `weight_decay=1e-4`, `dropout=0.5`, 2 epochs (vs. random-chance ≈1.1% at 90 classes — a 2.8× lift over chance, not yet a useful classifier).
- **Captured test accuracy:** **0.37%** under the same constraints.
- **~11.5 hours** of total assignment effort.

> The captured numbers reflect a CPU run on the partial Drive-uploaded subset (~1,800 rows) and a 2-epoch budget per trial. They are the floor, not the ceiling — the README's "Future Improvements" section lists the specific changes (full data, GPU, 20–30 epochs) expected to move the model out of the underfit regime.

## Context

Submitted as **Homework 2, Option 2** for **Harvard Extension School CSCI E-82 (Advanced Machine Learning), Fall 2025**. The assignment's stated objective: investigate whether the limiting factor in art-genre classification is the *number* of classes or the *number of examples per class*, and design experiments that produce evidence for the answer.
