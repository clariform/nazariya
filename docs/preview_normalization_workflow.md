# Preview normalization workflow

This workflow prepares normalized JPEG previews from RAW files so Nazariya can compare photos more fairly.

The goal is not to create final-looking edits. The goal is to remove enough exposure and white balance variation that similar photos produce useful feature embeddings.

## Inputs

Primary candidate CSV:

```text
data/inputs/candidates.csv
```

Working sampled CSV:

```text
data/inputs/candidates_sample_003_seed_42_swapped.csv
```

Per-candidate normalization settings:

```text
data/config/candidate_overrides.csv
```

The sampled CSV is usually created from the full candidate CSV. Keep the original sample if you want, but use one working swapped file for iterative review.

## Step 1: Create or refresh the override CSV

```bash
./scripts/nazariya make-overrides-template \
  --input data/inputs/candidates.csv \
  --output data/config/candidate_overrides.csv
```

The override CSV has one row per candidate set.

Columns:

```csv
candidate_key,wb,exposure_mode,target_median,low_pct,high_pct,user_wb,notes
```

Example:

```csv
candidate_key,wb,exposure_mode,target_median,low_pct,high_pct,user_wb,notes
c001,daylight,percentile,,0.5,99.5,,default
c011,daylight,center-midtone,0.18,,,,darken this set
```

## White balance modes

Supported values:

```text
daylight
camera
auto
gray-world
custom
```

Recommended default for search consistency:

```text
daylight
```

Use `custom` only when you want explicit raw channel multipliers.

Example:

```csv
c014,custom,center-midtone,0.38,,,2.0,1.0,1.45,1.0,custom WB test
```

The `user_wb` field is four raw multipliers:

```text
R,G1,B,G2
```

For normal tuning, prefer `daylight` unless a specific set clearly needs special handling.

## Exposure modes

Supported values:

```text
percentile
midtone
center-midtone
```

### percentile

Uses `low_pct` and `high_pct` to stretch luminance globally.

Good for:

```text
landscapes
architecture
wide scenes
flat objects
non-centered subjects
```

Default:

```text
low_pct  = 0.5
high_pct = 99.5
```

To make percentile previews darker, move `high_pct` closer to the true highlight limit, for example:

```text
99.7
99.9
```

To make percentile previews brighter, lower `high_pct`, for example:

```text
99.0
98.5
```

### midtone

Finds useful midtones while ignoring deep shadows and strong highlights, then scales the image toward `target_median`.

Good for:

```text
general scenes
foliage
environmental portraits
mixed scenes where center weighting is not ideal
```

### center-midtone

Like `midtone`, but gives more weight to the center of the image.

Good for:

```text
portraits
centered subjects
backlit portraits
images where edges/background should not dominate exposure
```

## Typical target_median values

`target_median` affects only:

```text
midtone
center-midtone
```

Useful values:

```text
0.18    very dark, useful for preserving heavy backlight mood
0.26    dark
0.30    clearly darker than default
0.34    slightly darker
0.38    balanced default
0.42    brighter
0.45    bright
0.50    usually too bright for search previews
```

Lower values darken the preview. Higher values brighten it.

## Step 2: Build a sampled candidate CSV

```bash
./scripts/nazariya sample \
  --input data/inputs/candidates.csv \
  --output data/inputs/candidates_sample_003_seed_42.csv \
  --per-candidate 3 \
  --seed 42
```

Recommended sample sizes:

```text
1 per candidate = smoke test
3 per candidate = normalization review
full CSV         = later, after settings are stable
```

## Step 3: Build previews

Write previews to a local scratch folder when possible. Network volumes can time out during long jobs.

```bash
LOCAL_PREVIEW_ROOT="$HOME/Documents/Scratch/tmp/Nazariya/previews/sample_003_seed_42_review"

caffeinate -dimsu ./scripts/nazariya build-previews \
  --input data/inputs/candidates_sample_003_seed_42_swapped.csv \
  --output "$LOCAL_PREVIEW_ROOT" \
  --max-size 768 \
  --overrides data/config/candidate_overrides.csv \
  --overwrite
```

The command writes:

```text
normalized/
debug_original/
preview_map.csv
failures.csv
```

## Step 4: Generate normalization contact sheets

```bash
LOCAL_SHEETS_ROOT="$HOME/Documents/Scratch/tmp/Nazariya/contact_sheets/sample_003_seed_42_review"

./scripts/nazariya contact-sheets \
  --preview-map "$LOCAL_PREVIEW_ROOT/preview_map.csv" \
  --output "$LOCAL_SHEETS_ROOT" \
  --thumb-size 420 \
  --columns 3

open "$LOCAL_SHEETS_ROOT"
```

Inspect each candidate set. You are looking for previews that are comparable enough for feature extraction, not final edits.

## Step 5: Tune one candidate set

Build only one candidate while tuning:

```bash
./scripts/nazariya build-previews \
  --input data/inputs/candidates_sample_003_seed_42_swapped.csv \
  --output "$HOME/Documents/Scratch/tmp/Nazariya/previews/tune_c011" \
  --candidate c011 \
  --max-size 768 \
  --overrides data/config/candidate_overrides.csv \
  --overwrite
```

Create a contact sheet for just that candidate:

```bash
./scripts/nazariya contact-sheets \
  --preview-map "$HOME/Documents/Scratch/tmp/Nazariya/previews/tune_c011/preview_map.csv" \
  --output "$HOME/Documents/Scratch/tmp/Nazariya/contact_sheets/tune_c011" \
  --thumb-size 420 \
  --columns 3

open "$HOME/Documents/Scratch/tmp/Nazariya/contact_sheets/tune_c011"
```

Edit `data/config/candidate_overrides.csv`, then rerun the same commands.

## Step 6: Swap a bad random sample

Sometimes the random sample picks a frame that does not represent the set, such as an image with a huge light source.

List sampled files for a candidate:

```bash
python - <<'PY'
import csv
from pathlib import Path

candidate = "c011"
p = Path("data/inputs/candidates_sample_003_seed_42_swapped.csv")

with p.open(newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("primary_candidate_key") == candidate:
            print(r.get("file_name"), r.get("source_path"))
PY
```

Swap one sampled row with another image from the same candidate:

```bash
./scripts/nazariya swap-sample \
  --full data/inputs/candidates.csv \
  --sample data/inputs/candidates_sample_003_seed_42_swapped.csv \
  --output data/inputs/candidates_sample_003_seed_42_swapped.csv \
  --candidate c011 \
  --remove-file BAD_FILE_NAME.ARW \
  --seed 99
```

The same file can be used as the working swapped sample to avoid creating too many versions.

Keep one backup before repeated swaps:

```bash
cp data/inputs/candidates_sample_003_seed_42_swapped.csv \
   data/inputs/candidates_sample_003_seed_42_swapped.backup.csv
```

## Step 7: Rebuild all review previews

Once overrides and swaps look good:

```bash
LOCAL_PREVIEW_ROOT="$HOME/Documents/Scratch/tmp/Nazariya/previews/sample_003_seed_42_review"

caffeinate -dimsu ./scripts/nazariya build-previews \
  --input data/inputs/candidates_sample_003_seed_42_swapped.csv \
  --output "$LOCAL_PREVIEW_ROOT" \
  --max-size 768 \
  --overrides data/config/candidate_overrides.csv \
  --overwrite
```

Then regenerate contact sheets:

```bash
LOCAL_SHEETS_ROOT="$HOME/Documents/Scratch/tmp/Nazariya/contact_sheets/sample_003_seed_42_review"

./scripts/nazariya contact-sheets \
  --preview-map "$LOCAL_PREVIEW_ROOT/preview_map.csv" \
  --output "$LOCAL_SHEETS_ROOT" \
  --thumb-size 420 \
  --columns 3
```

## Optional: Copy final previews to dataset storage

```bash
mkdir -p "$WHISK_ML_DATASETS/nazariya/previews"
mkdir -p "$WHISK_ML_DATASETS/nazariya/contact_sheets"

rsync -avh --progress \
  "$LOCAL_PREVIEW_ROOT/" \
  "$WHISK_ML_DATASETS/nazariya/previews/sample_003_seed_42_review/"

rsync -avh --progress \
  "$LOCAL_SHEETS_ROOT/" \
  "$WHISK_ML_DATASETS/nazariya/contact_sheets/sample_003_seed_42_review/"
```

## Notes

- Use `percentile` for broad global consistency.
- Use `center-midtone` for difficult portrait sets.
- Use lower `target_median` to darken `center-midtone` or `midtone` previews.
- Use `--candidate c###` for fast iteration.
- The preview settings are analysis-only. They do not modify Lightroom edits or the original RAW files.
