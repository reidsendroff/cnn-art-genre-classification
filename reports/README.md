# Reports

Static, shareable exports of the analysis notebook for reviewers who do not want to install Python.

## Expected files

| File                                   | Status        | Contents                                                                |
|----------------------------------------|---------------|-------------------------------------------------------------------------|
| `SendroffReidHW2.pdf`                  | TODO — generate | Full-notebook PDF including code, output, plots. Course submission format. |
| `SendroffReidHW2.html`                 | TODO — generate | Self-contained HTML render (better for hyperlinks, no dependencies)     |

## How to generate

### PDF via `nbconvert` (recommended for the course submission)

```bash
# from repo root, with the venv from requirements.txt active
pip install nbconvert pyppeteer
jupyter nbconvert \
    --to webpdf \
    --allow-chromium-download \
    --output reports/SendroffReidHW2.pdf \
    notebooks/SendroffReidHW2.ipynb
```

`webpdf` uses a headless Chromium so MathJax and styled markdown render correctly. If `pyppeteer` is awkward to install, fall back to:

```bash
jupyter nbconvert --to pdf --output reports/SendroffReidHW2.pdf notebooks/SendroffReidHW2.ipynb
```

That path requires a working LaTeX install (TeX Live or MacTeX). On macOS:

```bash
brew install --cask mactex-no-gui
```

### HTML (lightweight)

```bash
jupyter nbconvert \
    --to html \
    --output reports/SendroffReidHW2.html \
    notebooks/SendroffReidHW2.ipynb
```

Embeds all output cells inline; opens in any browser with no dependencies.

### Browser print (zero-install fallback)

1. Open the notebook in JupyterLab.
2. File → Save and Export Notebook As → PDF.
3. If that fails, `File → Print → Save as PDF` from the browser produces an acceptable document for the course portal.

## Usage notes

- Regenerate the PDF/HTML **after** every full notebook re-run so the embedded figures and metrics stay in sync with the code.
- Do not commit `*.pdf` or `*.html` files larger than 10 MB. The current notebook with all plots inline is well under that, but a re-run with high-resolution confusion matrices could push past it; if so, store on Drive and link from the top-level README instead of committing.
