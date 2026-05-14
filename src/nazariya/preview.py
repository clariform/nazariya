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


def parse_user_wb(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None

    parts = [p.strip() for p in value.split(",")]

    if len(parts) != 4:
        raise ValueError("--user-wb must be four comma-separated values, like 2.0,1.0,1.4,1.0")

    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError("--user-wb values must be numbers") from exc


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

    # daylight mode:
    # use_camera_wb=False, use_auto_wb=False, user_wb=None
    # LibRaw/rawpy uses its fixed/daylight-style white balance path.
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
    """
    Per-image gray-world white balance.

    Useful for experiments, but not ideal as the default because it can make
    images from the same scene drift differently based on framing/content.
    """
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


def exposure_normalize(
    img: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Percentile-based tonal normalization.

    This does not use Lightroom edits. It only makes previews easier to compare.
    """
    y = luminance(img)

    lo = float(np.percentile(y, low_pct))
    hi = float(np.percentile(y, high_pct))

    if hi <= lo + eps:
        return np.clip(img, 0.0, 1.0)

    out = (img - lo) / (hi - lo)

    over = np.maximum(out - 1.0, 0.0)
    out = np.where(out > 1.0, 1.0 + over / (1.0 + over), out)

    return np.clip(out, 0.0, 1.0)


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
    low_pct: float,
    high_pct: float,
    wb_mode: str,
) -> np.ndarray:
    out = img

    # Only apply content-dependent WB when explicitly requested.
    if wb_mode == "gray-world":
        out = gray_world_wb(out)

    out = exposure_normalize(out, low_pct=low_pct, high_pct=high_pct)
    out = resize_long_edge(out, max_size=max_size)
    return out


def debug_original_preview(img: np.ndarray, max_size: int) -> np.ndarray:
    """
    Basic viewable preview from the RAW render before Nazariya normalization.
    """
    out = exposure_normalize(img, low_pct=0.1, high_pct=99.9)
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
    overwrite: bool = False,
) -> PreviewBuildResult:
    if wb_mode not in VALID_WB_MODES:
        raise ValueError(f"--wb must be one of: {', '.join(sorted(VALID_WB_MODES))}")

    user_wb = parse_user_wb(user_wb_text)

    _fieldnames, rows = read_csv_rows(input_path)

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

        image_id = stable_id(str(source_path))
        normalized_path = normalized_dir / f"{image_id}.jpg"
        debug_path = debug_dir / f"{image_id}.jpg"

        if normalized_path.exists() and debug_path.exists() and not overwrite:
            skipped_existing += 1
        else:
            try:
                rgb = read_raw_rgb(
                    source_path,
                    wb_mode=wb_mode if wb_mode != "gray-world" else "daylight",
                    user_wb=user_wb,
                )

                debug_img = debug_original_preview(rgb, max_size=max_size)
                norm_img = normalize_for_search(
                    rgb,
                    max_size=max_size,
                    low_pct=low_pct,
                    high_pct=high_pct,
                    wb_mode=wb_mode,
                )

                save_jpeg(debug_path, debug_img)
                save_jpeg(normalized_path, norm_img)

                rendered += 1
            except Exception as exc:
                failed += 1
                failure_rows.append({
                    "row": str(index),
                    "source_path": str(source_path),
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
            "wb_mode": wb_mode,
            "max_size": str(max_size),
            "low_pct": str(low_pct),
            "high_pct": str(high_pct),
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
            "max_size",
            "low_pct",
            "high_pct",
        ],
        map_rows,
    )

    write_csv(
        failures_path,
        ["row", "source_path", "error"],
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
    )
