from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class FeatureExtractResult:
    preview_map_path: Path
    output_npz_path: Path
    output_metadata_path: Path
    total_rows: int
    extracted: int
    failed: int
    model_name: str
    pretrained: str
    feature_dim: int
    clip_dim: int
    color_dim: int


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


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def load_rgb_uint8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def center_weight_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    x = (xx - cx) / max(w, 1)
    y = (yy - cy) / max(h, 1)

    r2 = x * x + y * y
    mask = np.exp(-r2 / 0.08).astype(np.float32)
    return mask / max(float(mask.max()), 1e-6)


def edge_weight_mask(h: int, w: int) -> np.ndarray:
    center = center_weight_mask(h, w)
    edge = 1.0 - center
    return np.clip(edge, 0.0, 1.0).astype(np.float32)


def weighted_hist(
    values: np.ndarray,
    *,
    bins: int,
    value_range: tuple[float, float],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    values_flat = values.reshape(-1)

    if weights is not None:
        weights_flat = weights.reshape(-1).astype(np.float32)
    else:
        weights_flat = None

    hist, _ = np.histogram(
        values_flat,
        bins=bins,
        range=value_range,
        weights=weights_flat,
    )

    hist = hist.astype(np.float32)
    total = float(hist.sum())

    if total > 0:
        hist /= total

    return hist


def weighted_stats(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = values.astype(np.float32).reshape(-1)

    if weights is None:
        if values.size == 0:
            return np.zeros(4, dtype=np.float32)

        return np.array(
            [
                float(values.mean()),
                float(values.std()),
                float(np.percentile(values, 10)),
                float(np.percentile(values, 90)),
            ],
            dtype=np.float32,
        )

    weights = weights.astype(np.float32).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if int(valid.sum()) == 0:
        return np.zeros(4, dtype=np.float32)

    v = values[valid]
    w = weights[valid]
    w_sum = float(w.sum())

    if w_sum <= 0:
        return np.zeros(4, dtype=np.float32)

    mean = float((v * w).sum() / w_sum)
    var = float(((v - mean) ** 2 * w).sum() / w_sum)

    order = np.argsort(v)
    sv = v[order]
    sw = w[order]
    cdf = np.cumsum(sw) / max(float(sw.sum()), 1e-6)

    p10 = float(sv[min(int(np.searchsorted(cdf, 0.10)), len(sv) - 1)])
    p90 = float(sv[min(int(np.searchsorted(cdf, 0.90)), len(sv) - 1)])

    return np.array([mean, var ** 0.5, p10, p90], dtype=np.float32)


def masked_mean_std(channels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    channels: H x W x C float
    mask: H x W bool/float
    """
    mask = mask.astype(bool)

    if int(mask.sum()) < 20:
        return np.zeros(channels.shape[-1] * 2, dtype=np.float32)

    pixels = channels[mask]
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)

    return np.concatenate([mean, std]).astype(np.float32)


def region_color_hist_features(
    lab: np.ndarray,
    hsv: np.ndarray,
    *,
    weights: np.ndarray | None,
    prefix_scale: float = 1.0,
) -> np.ndarray:
    """
    Histograms for a specific region/mask.
    LAB OpenCV ranges:
      L: 0..255
      a: 0..255
      b: 0..255
    HSV OpenCV ranges:
      H: 0..179
      S: 0..255
      V: 0..255
    """
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    feats = [
        weighted_hist(l, bins=32, value_range=(0, 256), weights=weights),
        weighted_hist(a, bins=24, value_range=(0, 256), weights=weights),
        weighted_hist(b, bins=24, value_range=(0, 256), weights=weights),
        weighted_hist(h, bins=24, value_range=(0, 180), weights=weights),
        weighted_hist(s, bins=24, value_range=(0, 256), weights=weights),
        weighted_hist(v, bins=32, value_range=(0, 256), weights=weights),
    ]

    # Add compact stats for L/S/V. These are useful for editability.
    lf = l.astype(np.float32) / 255.0
    sf = s.astype(np.float32) / 255.0
    vf = v.astype(np.float32) / 255.0

    feats.extend([
        weighted_stats(lf, weights),
        weighted_stats(sf, weights),
        weighted_stats(vf, weights),
    ])

    out = np.concatenate(feats).astype(np.float32)
    return out * float(prefix_scale)


def tonal_zone_features(lab: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """
    Split pixels into shadows/midtones/highlights by luminance/value,
    then measure average color in each zone.

    This helps separate:
      cool shadows + warm highlights
      green shadows + neutral highlights
      high-key pastel scenes
    """
    labf = lab.astype(np.float32) / 255.0
    hsvf = hsv.astype(np.float32)
    hsvf[..., 0] /= 179.0
    hsvf[..., 1] /= 255.0
    hsvf[..., 2] /= 255.0

    l = labf[..., 0]

    shadows = l < 0.33
    midtones = (l >= 0.33) & (l < 0.72)
    highlights = l >= 0.72

    feats = [
        masked_mean_std(labf, shadows),
        masked_mean_std(labf, midtones),
        masked_mean_std(labf, highlights),
        masked_mean_std(hsvf, shadows),
        masked_mean_std(hsvf, midtones),
        masked_mean_std(hsvf, highlights),
        np.array(
            [
                float(shadows.mean()),
                float(midtones.mean()),
                float(highlights.mean()),
            ],
            dtype=np.float32,
        ),
    ]

    return np.concatenate(feats).astype(np.float32)


def bright_region_features(lab: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """
    Approximate where the brightest/most dominant light lives.

    Useful for:
      sunset backlight
      bright sky on one side
      center subject vs bright background
    """
    h_img, w_img = lab.shape[:2]
    l = lab[..., 0].astype(np.float32) / 255.0
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0

    threshold = float(np.percentile(l, 90))
    bright = l >= threshold

    if int(bright.sum()) < 20:
        return np.zeros(14, dtype=np.float32)

    yy, xx = np.mgrid[0:h_img, 0:w_img]
    x_norm = xx.astype(np.float32) / max(w_img - 1, 1)
    y_norm = yy.astype(np.float32) / max(h_img - 1, 1)

    weights = l * bright.astype(np.float32)
    w_sum = float(weights.sum())

    if w_sum <= 0:
        return np.zeros(14, dtype=np.float32)

    cx = float((x_norm * weights).sum() / w_sum)
    cy = float((y_norm * weights).sum() / w_sum)

    left = float((weights[:, : w_img // 3].sum()) / w_sum)
    center = float((weights[:, w_img // 3 : 2 * w_img // 3].sum()) / w_sum)
    right = float((weights[:, 2 * w_img // 3 :].sum()) / w_sum)

    top = float((weights[: h_img // 3, :].sum()) / w_sum)
    middle = float((weights[h_img // 3 : 2 * h_img // 3, :].sum()) / w_sum)
    bottom = float((weights[2 * h_img // 3 :, :].sum()) / w_sum)

    bright_ratio = float(bright.mean())
    bright_l_mean = float(l[bright].mean())
    bright_s_mean = float(s[bright].mean())
    bright_v_mean = float(v[bright].mean())

    center_mask = center_weight_mask(h_img, w_img)
    edge_mask = edge_weight_mask(h_img, w_img)

    center_l = float((l * center_mask).sum() / max(float(center_mask.sum()), 1e-6))
    edge_l = float((l * edge_mask).sum() / max(float(edge_mask.sum()), 1e-6))
    center_edge_delta = center_l - edge_l

    return np.array(
        [
            cx,
            cy,
            left,
            center,
            right,
            top,
            middle,
            bottom,
            bright_ratio,
            bright_l_mean,
            bright_s_mean,
            bright_v_mean,
            center_l,
            center_edge_delta,
        ],
        dtype=np.float32,
    )


def environment_bucket_features(lab: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """
    Rough color/environment buckets.

    These are intentionally heuristic. They help rank broad edit families:
      foliage
      ocean/sky
      beach/sand/earth
      sunset/warm
      neutral/gray
      low saturation/high key
    """
    h = hsv[..., 0].astype(np.float32)  # 0..179
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    l = lab[..., 0].astype(np.float32) / 255.0

    center = center_weight_mask(*h.shape)
    edge = edge_weight_mask(*h.shape)

    def ratio(mask: np.ndarray, weights: np.ndarray | None = None) -> float:
        m = mask.astype(np.float32)

        if weights is not None:
            m = m * weights.astype(np.float32)
            denom = float(weights.sum())
        else:
            denom = float(mask.size)

        if denom <= 0:
            return 0.0

        return float(m.sum() / denom)

    # OpenCV hue rough ranges.
    # red wraps around 0, yellow/orange ~10..35, green ~35..85, cyan/blue ~85..130.
    warm = ((h <= 25) | (h >= 165)) & (s > 0.20) & (v > 0.25)
    yellow_orange = (h > 10) & (h < 38) & (s > 0.18) & (v > 0.25)
    green = (h >= 35) & (h <= 90) & (s > 0.16) & (v > 0.18)
    cyan_blue = (h >= 85) & (h <= 130) & (s > 0.12) & (v > 0.20)
    magenta_red = ((h >= 130) | (h <= 8)) & (s > 0.18) & (v > 0.20)

    low_sat = s < 0.12
    high_key = l > 0.72
    dark = l < 0.25

    # Background weighted ratios matter more for environment.
    feats = np.array(
        [
            ratio(warm),
            ratio(yellow_orange),
            ratio(green),
            ratio(cyan_blue),
            ratio(magenta_red),
            ratio(low_sat),
            ratio(high_key),
            ratio(dark),

            ratio(warm, edge),
            ratio(yellow_orange, edge),
            ratio(green, edge),
            ratio(cyan_blue, edge),
            ratio(low_sat, edge),
            ratio(high_key, edge),

            ratio(warm, center),
            ratio(green, center),
            ratio(cyan_blue, center),
            ratio(low_sat, center),
            ratio(high_key, center),
        ],
        dtype=np.float32,
    )

    return feats


def color_light_features(rgb: np.ndarray) -> np.ndarray:
    """
    Editing-aware color/light feature.

    Designed for grouping images that may accept similar human-guided edits:
      - foliage backgrounds
      - ocean/beach scenes
      - sunset/backlit scenes
      - soft neutral/studio-ish scenes
      - similar shadow/midtone/highlight color behavior
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h_img, w_img = rgb.shape[:2]
    center_mask = center_weight_mask(h_img, w_img)
    edge_mask = edge_weight_mask(h_img, w_img)

    feats: list[np.ndarray] = []

    # Global frame: overall palette and tone.
    feats.append(region_color_hist_features(lab, hsv, weights=None, prefix_scale=1.0))

    # Center-ish region: likely subject/primary area.
    feats.append(region_color_hist_features(lab, hsv, weights=center_mask, prefix_scale=0.85))

    # Edge/background region: likely environment/background.
    # This is key for foliage/ocean/beach/sunset grouping.
    feats.append(region_color_hist_features(lab, hsv, weights=edge_mask, prefix_scale=1.15))

    # Shadow/midtone/highlight behavior.
    feats.append(tonal_zone_features(lab, hsv) * 1.15)

    # Bright region / backlight proxy.
    feats.append(bright_region_features(lab, hsv) * 1.20)

    # Rough semantic color-environment buckets.
    feats.append(environment_bucket_features(lab, hsv) * 1.35)

    out = np.concatenate(feats).astype(np.float32)
    out = l2_normalize(out.reshape(1, -1))[0]
    return out


def device_for_torch() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def extract_features(
    *,
    preview_map_path: Path,
    output_npz_path: Path,
    output_metadata_path: Path,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    clip_weight: float = 0.65,
    color_weight: float = 0.35,
    batch_size: int = 32,
) -> FeatureExtractResult:
    import open_clip

    if clip_weight < 0 or color_weight < 0:
        raise ValueError("clip_weight and color_weight must be >= 0")

    if clip_weight == 0 and color_weight == 0:
        raise ValueError("At least one feature weight must be greater than 0")

    _fieldnames, rows = read_csv_rows(preview_map_path)

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    device = device_for_torch()

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model.eval()
    model.to(device)

    valid_rows: list[dict[str, str]] = []
    pil_images: list[Image.Image] = []
    color_feats: list[np.ndarray] = []
    failure_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        preview_path = Path(str(row.get("normalized_preview_path", "")).strip())

        if not preview_path.exists():
            failure_rows.append({
                "row": str(idx),
                "image_id": str(row.get("image_id", "")),
                "source_path": str(row.get("source_path", "")),
                "error": f"missing preview: {preview_path}",
            })
            continue

        try:
            rgb = load_rgb_uint8(preview_path)
            pil_img = Image.fromarray(rgb, mode="RGB")
            cf = color_light_features(rgb)

            valid_rows.append(row)
            pil_images.append(pil_img)
            color_feats.append(cf)
        except Exception as exc:
            failure_rows.append({
                "row": str(idx),
                "image_id": str(row.get("image_id", "")),
                "source_path": str(row.get("source_path", "")),
                "error": repr(exc),
            })

    if not valid_rows:
        raise RuntimeError("No valid preview rows found. Cannot extract features.")

    clip_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(pil_images), batch_size):
            batch_imgs = pil_images[start:start + batch_size]
            batch = torch.stack([preprocess(img) for img in batch_imgs]).to(device)

            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            clip_chunks.append(emb.detach().cpu().numpy().astype(np.float32))

    clip_features = np.concatenate(clip_chunks, axis=0).astype(np.float32)
    color_features = np.stack(color_feats, axis=0).astype(np.float32)

    clip_features = l2_normalize(clip_features)
    color_features = l2_normalize(color_features)

    combined = np.concatenate(
        [
            clip_features * float(clip_weight),
            color_features * float(color_weight),
        ],
        axis=1,
    ).astype(np.float32)

    combined = l2_normalize(combined)

    image_ids = np.array([str(r.get("image_id", "")) for r in valid_rows])
    source_paths = np.array([str(r.get("source_path", "")) for r in valid_rows])
    candidate_keys = np.array([str(r.get("primary_candidate_key", "")) for r in valid_rows])
    preview_paths = np.array([str(r.get("normalized_preview_path", "")) for r in valid_rows])

    np.savez_compressed(
        output_npz_path,
        image_ids=image_ids,
        source_paths=source_paths,
        candidate_keys=candidate_keys,
        preview_paths=preview_paths,
        features=combined,
        clip_features=clip_features,
        color_features=color_features,
        clip_weight=np.array([clip_weight], dtype=np.float32),
        color_weight=np.array([color_weight], dtype=np.float32),
        model_name=np.array([model_name]),
        pretrained=np.array([pretrained]),
        color_feature_version=np.array(["editing-aware-v2"]),
    )

    metadata_fields = [
        "feature_index",
        "image_id",
        "primary_candidate_key",
        "candidate_keys",
        "source_path",
        "normalized_preview_path",
        "debug_original_preview_path",
        "file_name",
        "file_stem",
        "capture_time",
        "wb_mode",
        "exposure_mode",
        "target_median",
        "low_pct",
        "high_pct",
        "override_applied",
        "override_note",
    ]

    metadata_rows: list[dict[str, str]] = []
    for i, row in enumerate(valid_rows):
        out = dict(row)
        out["feature_index"] = str(i)
        metadata_rows.append(out)

    write_csv(output_metadata_path, metadata_fields, metadata_rows)

    if failure_rows:
        failures_path = output_metadata_path.with_name(output_metadata_path.stem + "_failures.csv")
        write_csv(
            failures_path,
            ["row", "image_id", "source_path", "error"],
            failure_rows,
        )

    return FeatureExtractResult(
        preview_map_path=preview_map_path,
        output_npz_path=output_npz_path,
        output_metadata_path=output_metadata_path,
        total_rows=len(rows),
        extracted=len(valid_rows),
        failed=len(failure_rows),
        model_name=model_name,
        pretrained=pretrained,
        feature_dim=int(combined.shape[1]),
        clip_dim=int(clip_features.shape[1]),
        color_dim=int(color_features.shape[1]),
    )
