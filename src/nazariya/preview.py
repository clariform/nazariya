from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image


VALID_WB_MODES = {"daylight", "camera", "auto", "gray-world", "custom"}
VALID_EXPOSURE_MODES = {"percentile", "midtone", "center-midtone"}


@dataclass(frozen=True)
class PreviewSettings:
    wb_mode: str
    exposure_mode: str
    target_median: float
    low_pct: float
    high_pct: float
    user_wb_text: str | None = None
    override_note: str = ""


@dataclass(frozen=True)
class PreviewBuildResult:
    input_path: Path
    output_root: Path
    total_rows: int
    rendered: int
    skipped_existing: int
    failed: int
    preview_map_path: Path
    failures_path: Path
    wb_mode: str
    exposure_mode: str
    overrides_path: Path | None
    overrides_loaded: int
    candidate_filter: str | None


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float_or_default(value: str | None, default: float) -> float:
    text = str(value or "").strip()

    if not text:
        return default

    return float(text)


def parse_user_wb(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    parts = [p.strip() for p in text.split(",")]

    if len(parts) != 4:
        raise ValueError("--user-wb must be four comma-separated values, like 2.0,1.0,1.4,1.0")

    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError("--user-wb values must be numbers") from exc


def load_candidate_overrides(
    overrides_path: Path | None,
    defaults: PreviewSettings,
) -> dict[str, PreviewSettings]:
    if overrides_path is None:
        return {}

    if not overrides_path.exists():
        raise FileNotFoundError(f"Override CSV not found: {overrides_path}")

    _fieldnames, rows = read_csv_rows(overrides_path)
    out: dict[str, PreviewSettings] = {}

    for idx, row in enumerate(rows, start=2):
        candidate_key = str(row.get("candidate_key", "")).strip()

        if not candidate_key:
            continue

        wb_mode = str(row.get("wb", "")).strip() or defaults.wb_mode
        exposure_mode = str(row.get("exposure_mode", "")).strip() or defaults.exposure_mode

        if wb_mode not in VALID_WB_MODES:
            raise ValueError(
                f"{overrides_path}:{idx}: invalid wb {wb_mode!r}. "
                f"Valid: {sorted(VALID_WB_MODES)}"
            )

        if exposure_mode not in VALID_EXPOSURE_MODES:
            raise ValueError(
                f"{overrides_path}:{idx}: invalid exposure_mode {exposure_mode!r}. "
                f"Valid: {sorted(VALID_EXPOSURE_MODES)}"
            )

        out[candidate_key] = PreviewSettings(
            wb_mode=wb_mode,
            exposure_mode=exposure_mode,
            target_median=parse_float_or_default(row.get("target_median"), defaults.target_median),
            low_pct=parse_float_or_default(row.get("low_pct"), defaults.low_pct),
            high_pct=parse_float_or_default(row.get("high_pct"), defaults.high_pct),
            user_wb_text=(str(row.get("user_wb", "")).strip() or defaults.user_wb_text),
            override_note=str(row.get("notes", "")).strip(),
        )

    return out


def resolve_settings(
    row: dict[str, str],
    defaults: PreviewSettings,
    overrides: dict[str, PreviewSettings],
) -> tuple[str, PreviewSettings, bool]:
    candidate_key = str(row.get("primary_candidate_key", "")).strip()

    if candidate_key in overrides:
        return candidate_key, overrides[candidate_key], True

    return candidate_key, defaults, False


def row_matches_candidate_filter(row: dict[str, str], candidate_filter: str | None) -> bool:
    if candidate_filter is None:
        return True

    wanted = candidate_filter.strip()
    if not wanted:
        return True

    primary = str(row.get("primary_candidate_key", "")).strip()
    if primary == wanted:
        return True

    keys_text = str(row.get("candidate_keys", ""))
    keys = [k.strip() for k in keys_text.split(";") if k.strip()]
    return wanted in keys


def read_raw_rgb(
    path: Path,
    *,
    wb_mode: str = "daylight",
    user_wb: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """
    Read a RAW file and return RGB float32 image in 0..1.

    This is analysis-only. It deliberately ignores Lightroom edits and XMP.
    It reads captured RAW data through LibRaw/rawpy.
    """
    if wb_mode not in VALID_WB_MODES:
        raise ValueError(f"Invalid wb_mode: {wb_mode}. Valid modes: {sorted(VALID_WB_MODES)}")

    use_camera_wb = wb_mode == "camera"
    use_auto_wb = wb_mode == "auto"

    raw_user_wb = None
    if wb_mode == "custom":
        if user_wb is None:
            raise ValueError("--wb custom requires --user-wb R,G1,B,G2")
        raw_user_wb = user_wb

    with rawpy.imread(str(path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=use_camera_wb,
            use_auto_wb=use_auto_wb,
            user_wb=raw_user_wb,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
        )

    rgb = rgb16.astype(np.float32) / 65535.0
    return np.clip(rgb, 0.0, 1.0)


def resize_long_edge(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Image has invalid dimensions")

    scale = max_size / max(h, w)

    if scale >= 1.0:
        return img

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def gray_world_wb(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    means = img.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    scale = target / (means + eps)
    out = img * scale
    return np.clip(out, 0.0, 1.0)


def luminance(img: np.ndarray) -> np.ndarray:
    return (
        0.2126 * img[..., 0]
        + 0.7152 * img[..., 1]
        + 0.0722 * img[..., 2]
    )


def soft_clip_highlights(img: np.ndarray) -> np.ndarray:
    over = np.maximum(img - 1.0, 0.0)
    out = np.where(img > 1.0, 1.0 + over / (1.0 + over), img)
    return np.clip(out, 0.0, 1.0)


def exposure_normalize_percentile(
    img: np.ndarray,
    *,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    y = luminance(img)

    lo = float(np.percentile(y, low_pct))
    hi = float(np.percentile(y, high_pct))

    if hi <= lo + eps:
        return np.clip(img, 0.0, 1.0)

    out = (img - lo) / (hi - lo)
    return soft_clip_highlights(out)


def exposure_normalize_midtone(
    img: np.ndarray,
    *,
    target_median: float = 0.38,
    low_clip: float = 0.02,
    high_clip: float = 0.92,
    eps: float = 1e-6,
) -> np.ndarray:
    y = luminance(img)

    mask = (y > low_clip) & (y < high_clip)

    if int(mask.sum()) < 100:
        return exposure_normalize_percentile(img, low_pct=0.5, high_pct=99.5)

    current_median = float(np.median(y[mask]))

    if current_median <= eps:
        return np.clip(img, 0.0, 1.0)

    scale = target_median / current_median
    out = img * scale

    return soft_clip_highlights(out)


def center_weight_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    x = (xx - cx) / max(w, 1)
    y = (yy - cy) / max(h, 1)

    r2 = x * x + y * y
    return np.exp(-r2 / 0.08).astype(np.float32)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = values.reshape(-1)
    weights = weights.reshape(-1)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if int(valid.sum()) == 0:
        return float(np.median(values))

    values = values[valid]
    weights = weights[valid]

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cutoff = quantile * cumulative[-1]

    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    idx = min(max(idx, 0), len(values) - 1)

    return float(values[idx])


def exposure_normalize_center_midtone(
    img: np.ndarray,
    *,
    target_median: float = 0.38,
    low_clip: float = 0.02,
    high_clip: float = 0.92,
    eps: float = 1e-6,
) -> np.ndarray:
    y = luminance(img)
    h, w = y.shape[:2]

    center_weights = center_weight_mask(h, w)
    midtone_mask = (y > low_clip) & (y < high_clip)

    weights = np.where(midtone_mask, center_weights, 0.0).astype(np.float32)

    if int(np.count_nonzero(weights)) < 100:
        return exposure_normalize_midtone(
            img,
            target_median=target_median,
            low_clip=low_clip,
            high_clip=high_clip,
        )

    current_median = weighted_quantile(y, weights, 0.5)

    if current_median <= eps:
        return np.clip(img, 0.0, 1.0)

    scale = target_median / current_median
    out = img * scale

    return soft_clip_highlights(out)


def apply_exposure_normalization(
    img: np.ndarray,
    *,
    exposure_mode: str,
    low_pct: float,
    high_pct: float,
    target_median: float,
) -> np.ndarray:
    if exposure_mode == "percentile":
        return exposure_normalize_percentile(
            img,
            low_pct=low_pct,
            high_pct=high_pct,
        )

    if exposure_mode == "midtone":
        return exposure_normalize_midtone(
            img,
            target_median=target_median,
        )

    if exposure_mode == "center-midtone":
        return exposure_normalize_center_midtone(
            img,
            target_median=target_median,
        )

    raise ValueError(
        f"Invalid exposure_mode: {exposure_mode}. "
        f"Valid modes: {sorted(VALID_EXPOSURE_MODES)}"
    )


def to_viewable_srgb_uint8(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    img = np.power(img, 1.0 / 2.2)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def save_jpeg(path: Path, img: np.ndarray, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    img8 = to_viewable_srgb_uint8(img)
    Image.fromarray(img8, mode="RGB").save(path, "JPEG", quality=quality, optimize=True)


def normalize_for_search(
    img: np.ndarray,
    *,
    max_size: int,
    settings: PreviewSettings,
) -> np.ndarray:
    out = img

    if settings.wb_mode == "gray-world":
        out = gray_world_wb(out)

    out = apply_exposure_normalization(
        out,
        exposure_mode=settings.exposure_mode,
        low_pct=settings.low_pct,
        high_pct=settings.high_pct,
        target_median=settings.target_median,
    )

    out = resize_long_edge(out, max_size=max_size)
    return out


def debug_original_preview(img: np.ndarray, max_size: int) -> np.ndarray:
    out = exposure_normalize_percentile(img, low_pct=0.1, high_pct=99.9)
    out = resize_long_edge(out, max_size=max_size)
    return out


def build_previews(
    *,
    input_path: Path,
    output_root: Path,
    max_size: int = 768,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
    wb_mode: str = "daylight",
    user_wb_text: str | None = None,
    exposure_mode: str = "center-midtone",
    target_median: float = 0.38,
    overrides_path: Path | None = None,
    candidate_filter: str | None = None,
    overwrite: bool = False,
) -> PreviewBuildResult:
    if wb_mode not in VALID_WB_MODES:
        raise ValueError(f"--wb must be one of: {', '.join(sorted(VALID_WB_MODES))}")

    if exposure_mode not in VALID_EXPOSURE_MODES:
        raise ValueError(
            f"--exposure-mode must be one of: {', '.join(sorted(VALID_EXPOSURE_MODES))}"
        )

    defaults = PreviewSettings(
        wb_mode=wb_mode,
        exposure_mode=exposure_mode,
        target_median=target_median,
        low_pct=low_pct,
        high_pct=high_pct,
        user_wb_text=user_wb_text,
    )

    overrides = load_candidate_overrides(overrides_path, defaults)

    _fieldnames, rows_all = read_csv_rows(input_path)
    rows = [
        row
        for row in rows_all
        if row_matches_candidate_filter(row, candidate_filter)
    ]

    normalized_dir = output_root / "normalized"
    debug_dir = output_root / "debug_original"
    map_path = output_root / "preview_map.csv"
    failures_path = output_root / "failures.csv"

    normalized_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    map_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []

    rendered = 0
    skipped_existing = 0
    failed = 0

    for index, row in enumerate(rows, start=1):
        source_path_text = str(row.get("source_path", "")).strip()
        source_path = Path(source_path_text)

        if not source_path_text:
            failed += 1
            failure_rows.append({
                "row": str(index),
                "source_path": "",
                "error": "missing source_path",
            })
            continue

        candidate_key, settings, override_applied = resolve_settings(row, defaults, overrides)
        effective_raw_wb_mode = settings.wb_mode if settings.wb_mode != "gray-world" else "daylight"
        user_wb = parse_user_wb(settings.user_wb_text)

        image_id = stable_id(str(source_path))
        normalized_path = normalized_dir / f"{image_id}.jpg"
        debug_path = debug_dir / f"{image_id}.jpg"

        if normalized_path.exists() and debug_path.exists() and not overwrite:
            skipped_existing += 1
        else:
            try:
                rgb = read_raw_rgb(
                    source_path,
                    wb_mode=effective_raw_wb_mode,
                    user_wb=user_wb,
                )

                debug_img = debug_original_preview(rgb, max_size=max_size)
                norm_img = normalize_for_search(
                    rgb,
                    max_size=max_size,
                    settings=settings,
                )

                save_jpeg(debug_path, debug_img)
                save_jpeg(normalized_path, norm_img)

                rendered += 1
            except Exception as exc:
                failed += 1
                failure_rows.append({
                    "row": str(index),
                    "source_path": str(source_path),
                    "candidate_key": candidate_key,
                    "error": repr(exc),
                })
                continue

        map_rows.append({
            "image_id": image_id,
            "source_path": str(source_path),
            "normalized_preview_path": str(normalized_path),
            "debug_original_preview_path": str(debug_path),
            "primary_candidate_key": str(row.get("primary_candidate_key", "")),
            "candidate_keys": str(row.get("candidate_keys", "")),
            "file_name": str(row.get("file_name", "")),
            "file_stem": str(row.get("file_stem", "")),
            "capture_time": str(row.get("capture_time", "")),
            "wb_mode": settings.wb_mode,
            "exposure_mode": settings.exposure_mode,
            "target_median": str(settings.target_median),
            "low_pct": str(settings.low_pct),
            "high_pct": str(settings.high_pct),
            "user_wb": settings.user_wb_text or "",
            "override_applied": "true" if override_applied else "false",
            "override_note": settings.override_note,
            "candidate_filter": candidate_filter or "",
            "max_size": str(max_size),
        })

    write_csv(
        map_path,
        [
            "image_id",
            "source_path",
            "normalized_preview_path",
            "debug_original_preview_path",
            "primary_candidate_key",
            "candidate_keys",
            "file_name",
            "file_stem",
            "capture_time",
            "wb_mode",
            "exposure_mode",
            "target_median",
            "low_pct",
            "high_pct",
            "user_wb",
            "override_applied",
            "override_note",
            "candidate_filter",
            "max_size",
        ],
        map_rows,
    )

    write_csv(
        failures_path,
        ["row", "source_path", "candidate_key", "error"],
        failure_rows,
    )

    return PreviewBuildResult(
        input_path=input_path,
        output_root=output_root,
        total_rows=len(rows),
        rendered=rendered,
        skipped_existing=skipped_existing,
        failed=failed,
        preview_map_path=map_path,
        failures_path=failures_path,
        wb_mode=wb_mode,
        exposure_mode=exposure_mode,
        overrides_path=overrides_path,
        overrides_loaded=len(overrides),
        candidate_filter=candidate_filter,
    )
