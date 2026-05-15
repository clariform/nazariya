# Nazariya

Nazariya is a local visual search and clustering tool for photo archives.

The current goal is to help find photos that are visually close enough to support a realistic style transfer with human input, but not so identical that simple copy-paste settings would be enough.

In practice, Nazariya helps answer questions like:

- Which candidate sets are visually close to each other?
- Which photos share similar lighting, palette, environment, or editability?
- Which sets should I inspect in Lightroom when building stronger delivery groups?

Nazariya currently works as a Python CLI, with a small Lightroom Classic helper plugin for exporting catalog metadata to CSV.

## Current workflow

### 1. Export candidate metadata from Lightroom Classic

Use the included Lightroom plugin to export a CSV of RAW/DNG photos that are in your candidate keyword sets.

The plugin exports metadata only. It does not export or copy the RAW files.

Example output:

```text
$MATRIX/packages/nazariya/data/inputs/candidates.csv
```

This CSV includes source paths, candidate keys, keywords, ratings, labels, capture time, camera/lens metadata, and location fields when Lightroom exposes them.

For this project, the current candidate set contains roughly 39,397 RAW photos across candidate keys such as `c001` through `c325`.

### 2. Create a per-candidate normalization override CSV

Create a CSV where each candidate set can define white balance and exposure normalization settings.

```bash
./scripts/nazariya make-overrides-template \
  --input data/inputs/candidates.csv \
  --output data/config/candidate_overrides.csv
```

Most rows can use the same default settings. Override only sets that need adjustment.

### 3. Build a smaller review sample

For fast iteration, sample a few images from each candidate set.

```bash
./scripts/nazariya sample \
  --input data/inputs/candidates.csv \
  --output data/inputs/candidates_sample_003_seed_42.csv \
  --per-candidate 3 \
  --seed 42
```

This produces a manageable visual review set. Three images per candidate is usually a good starting point because one image can be misleading.

### 4. Build normalized previews and inspect them

Nazariya reads the original RAW files, applies analysis-only white balance and exposure normalization, then writes JPEG previews.

The purpose is not to make the previews look beautiful. The purpose is to make visually comparable previews so similar images land close together in feature space.

This step is iterative:

1. Build previews.
2. Generate contact sheets.
3. Inspect exposure and white balance consistency.
4. Adjust `data/config/candidate_overrides.csv`.
5. Swap bad random samples when a picked frame does not represent the set.
6. Rebuild previews.

See the detailed workflow:

- [Preview normalization workflow](docs/preview_normalization_workflow.md)

### 5. Extract features and generate neighbor sheets

Once the previews look usable, generate embeddings and color/light features.

Nazariya currently combines:

- CLIP image embeddings
- LAB/HSV color and tonal histograms
- simple color/light statistics

For edit-family search, color and lighting usually matter more than semantic similarity, so a color-heavy weighting can be useful.

Example:

```bash
./scripts/nazariya extract-features \
  --preview-map "$WHISK_ML_DATASETS/nazariya/previews/sample_003_seed_42_review/preview_map.csv" \
  --output "$WHISK_ML_DATASETS/nazariya/features/sample_003_seed_42_review/features_clip035_color065.npz" \
  --metadata "$WHISK_ML_DATASETS/nazariya/features/sample_003_seed_42_review/features_clip035_color065.csv" \
  --clip-weight 0.35 \
  --color-weight 0.65 \
  --batch-size 32
```

Then generate one neighbor contact sheet per candidate set:

```bash
./scripts/nazariya neighbor-sheets \
  --features "$WHISK_ML_DATASETS/nazariya/features/sample_003_seed_42_review/features_clip035_color065.npz" \
  --preview-map "$WHISK_ML_DATASETS/nazariya/previews/sample_003_seed_42_review/preview_map.csv" \
  --output "$WHISK_ML_DATASETS/nazariya/debug_neighbors/by_candidate_clip035_color065" \
  --top-k 10 \
  --exclude-same-candidate \
  --thumb-size 260
```

See the detailed workflow:

- [Feature extraction and neighbor review workflow](docs/neighbor_review_workflow.md)

## Lightroom plugin

The Lightroom helper plugin lives here:

```text
lightroom/nazariya.lrplugin
```

Current menu item:

```text
Library > Plug-in Extras > Export Candidate CSV
```

It exports selected Lightroom photos to the configured CSV path in `lightroom/nazariya.lrplugin/Config.lua`.

## Data layout

Recommended local repo layout:

```text
data/
  config/
    candidate_overrides.csv
  inputs/
    candidates.csv
    candidates_backup.csv
    candidates_sample_003_seed_42.csv
    candidates_sample_003_seed_42_swapped.csv
  previews/
```

Generated previews, features, and contact sheets can be written to a larger dataset volume such as:

```text
$WHISK_ML_DATASETS/nazariya/
```

For long jobs, local scratch is often more reliable than a network volume. You can render locally, then copy results to the dataset volume afterward.

## CLI commands currently used

```bash
./scripts/nazariya --version
./scripts/nazariya hello
./scripts/nazariya init
./scripts/nazariya sample
./scripts/nazariya swap-sample
./scripts/nazariya make-overrides-template
./scripts/nazariya build-previews
./scripts/nazariya contact-sheets
./scripts/nazariya extract-features
./scripts/nazariya neighbor-sheets
```

## Development

This project uses `uv`.

```bash
uv sync
./scripts/nazariya --help
```

Build:

```bash
uv build
```

Publish:

```bash
uv publish
```

## Notes

Nazariya is still experimental. The current approach is intentionally practical:

1. Use Lightroom to identify candidate pools.
2. Use Python to normalize RAW previews and compute features.
3. Use contact sheets for human visual review.
4. Use nearest-neighbor suggestions to rebuild better photo groups manually.

The human review step is part of the design. The tool suggests promising neighborhoods; the final grouping still depends on taste, context, and the editing goal.
