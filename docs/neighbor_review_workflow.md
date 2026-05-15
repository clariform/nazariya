# Feature extraction and neighbor review workflow

This workflow uses normalized previews to find visually related candidate sets.

The goal is to identify photos and candidate sets that share editability: similar lighting, color palette, environment, tonal structure, or mood. These suggestions are meant to guide human review in Lightroom.

## Inputs

Normalized previews:

```text
$WHISK_ML_DATASETS/nazariya/previews/sample_003_seed_42_review/
```

Preview map:

```text
$WHISK_ML_DATASETS/nazariya/previews/sample_003_seed_42_review/preview_map.csv
```

The preview map links each generated preview back to:

```text
image_id
source_path
candidate key
file name
normalization settings
```

## Step 1: Extract features

Nazariya currently combines two feature blocks:

```text
CLIP embedding
color/light feature
```

The CLIP embedding helps with semantic and compositional similarity.

The color/light feature helps with editability: tone, palette, saturation, and luminance distribution.

For style-transfer grouping, color and lighting usually matter more than semantic similarity.

Recommended edit-family weighting:

```bash
./scripts/nazariya extract-features \
  --preview-map "$WHISK_ML_DATASETS/nazariya/previews/sample_003_seed_42_review/preview_map.csv" \
  --output "$WHISK_ML_DATASETS/nazariya/features/sample_003_seed_42_review/features_clip035_color065.npz" \
  --metadata "$WHISK_ML_DATASETS/nazariya/features/sample_003_seed_42_review/features_clip035_color065.csv" \
  --clip-weight 0.35 \
  --color-weight 0.65 \
  --batch-size 32
```

More color/light-heavy test:

```bash
./scripts/nazariya extract-features \
  --preview-map "$WHISK_ML_DATASETS/nazariya/previews/sample_003_seed_42_review/preview_map.csv" \
  --output "$WHISK_ML_DATASETS/nazariya/features/sample_003_seed_42_review/features_clip020_color080.npz" \
  --metadata "$WHISK_ML_DATASETS/nazariya/features/sample_003_seed_42_review/features_clip020_color080.csv" \
  --clip-weight 0.20 \
  --color-weight 0.80 \
  --batch-size 32
```

## Feature weighting guidance

### Semantic-heavy

```text
clip_weight  = 0.80
color_weight = 0.20
```

Finds similar subject, composition, and scene type.

### Balanced

```text
clip_weight  = 0.50
color_weight = 0.50
```

General-purpose visual similarity.

### Edit-lighting

```text
clip_weight  = 0.35
color_weight = 0.65
```

Recommended starting point for realistic style-transfer grouping.

### Color/environment-heavy

```text
clip_weight  = 0.20
color_weight = 0.80
```

Prioritizes lighting, palette, environment, and tonal family. Useful for discovering beach, foliage, sunset, urban, forest, studio, or similar edit families.

## Step 2: Generate neighbor sheets

Generate one contact sheet per candidate set.

Each sheet shows:

```text
query image on the left
top nearest images from other candidate sets to the right
similarity score
candidate key
file name
```

```bash
./scripts/nazariya neighbor-sheets \
  --features "$WHISK_ML_DATASETS/nazariya/features/sample_003_seed_42_review/features_clip035_color065.npz" \
  --preview-map "$WHISK_ML_DATASETS/nazariya/previews/sample_003_seed_42_review/preview_map.csv" \
  --output "$WHISK_ML_DATASETS/nazariya/debug_neighbors/by_candidate_clip035_color065" \
  --top-k 10 \
  --exclude-same-candidate \
  --thumb-size 260

open "$WHISK_ML_DATASETS/nazariya/debug_neighbors/by_candidate_clip035_color065"
```

For a more color-heavy pass:

```bash
./scripts/nazariya neighbor-sheets \
  --features "$WHISK_ML_DATASETS/nazariya/features/sample_003_seed_42_review/features_clip020_color080.npz" \
  --preview-map "$WHISK_ML_DATASETS/nazariya/previews/sample_003_seed_42_review/preview_map.csv" \
  --output "$WHISK_ML_DATASETS/nazariya/debug_neighbors/by_candidate_clip020_color080" \
  --top-k 10 \
  --exclude-same-candidate \
  --thumb-size 260

open "$WHISK_ML_DATASETS/nazariya/debug_neighbors/by_candidate_clip020_color080"
```

## Step 3: Compare feature weightings

Open the same candidate in multiple runs:

```bash
open "$WHISK_ML_DATASETS/nazariya/debug_neighbors/by_candidate_clip035_color065/c145.jpg"
open "$WHISK_ML_DATASETS/nazariya/debug_neighbors/by_candidate_clip020_color080/c145.jpg"
```

Judge the output by usefulness, not just score.

Useful neighbor suggestions should share some of:

```text
lighting direction or quality
background/environment color
shadow/highlight behavior
scene family
mood
tonal range
palette
```

They should not merely be:

```text
same model
same pose
same outfit
same semantic subject
```

For the current project, editability matters more than subject identity.

## Step 4: Inspect candidate neighborhoods

Use the sheets as a bird's-eye view.

Good candidates to mix are often sets where the neighbors feel like the same edit family, but not identical enough for copy-paste settings.

Look for groups such as:

```text
foliage / green ambient shade
beach / open sky / sand reflection
warm sunset backlight
cool overcast city
forest path / green cast
urban bridge / haze / backlight
high-key pale dress / soft environment
studio-neutral portrait sets
```

The output is not the final grouping. It is a map for deciding where to look in Lightroom.

## Step 5: Use results in Lightroom

When a candidate set has promising neighbors:

1. Open the candidate and neighbor sets in Lightroom.
2. Review the original RAWs, not only the normalized previews.
3. Build new delivery groups from compatible images.
4. Prefer groups where images can be made visually consistent with human adjustments, but would not be solved by a simple copy-paste edit.

## What the scores mean

The current neighbor sheet uses cosine similarity over the combined feature vector.

Higher score means closer according to the selected feature weighting.

However, a high score is not automatically a good grouping choice. Use visual judgment.

Examples:

```text
High score + same shoot + same pose:
  likely too similar

Medium-high score + similar light/environment + different scene details:
  potentially useful

High color score + unrelated subject/context:
  maybe too color-driven
```

## Recommended current setting

For the current style-transfer search:

```text
clip_weight  = 0.35
color_weight = 0.65
```

Also test:

```text
clip_weight  = 0.20
color_weight = 0.80
```

Use whichever produces better candidate neighborhoods for the project.

## Future improvements

Possible next steps:

```text
set-level neighbor ranking
candidate-pair contact sheets
penalties for same folder / same date / likely copy-paste risk
center-weighted color features
background color features
shadow/midtone/highlight color features
DINOv2 or other visual embeddings
Lightroom collection creation from neighbor results
```

The current command is intentionally simple: generate sheets, inspect visually, and use the results to guide Lightroom grouping.
