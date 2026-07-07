from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


DEFAULT_SEGMENT_MODEL = "briaai/RMBG-2.0"


@dataclass(frozen=True)
class BackgroundFeatureExtractResult:
    preview_map_path: Path
    output_npz_path: Path
    output_metadata_path: Path
    total_rows: int
    extracted: int
    failed: int
    clip_model_name: str
    clip_pretrained: str
    segment_model: str
    feature_dim: int
    clip_dim: int
    background_dim: int
    clip_weight: float
    background_weight: float
    debug_dir: Path | None


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


def device_for_torch() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def load_rgb_uint8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.uint8)


def load_segmentation_model(model_id: str) -> tuple[torch.nn.Module, transforms.Compose, str]:
    device = device_for_torch()

    model = AutoModelForImageSegmentation.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    return model, transform, device


def predict_subject_mask(
    *,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: str,
    image: Image.Image,
) -> np.ndarray:
    """
    Return subject alpha mask as float32 0..1 at the original image size.
    """
    original_size = image.size  # width, height
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(x)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil = pred_pil.resize(original_size, Image.Resampling.BILINEAR)

    mask = np.asarray(pred_pil).astype(np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0)


def center_weight_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    x = (xx - cx) / max(w, 1)
    y = (yy - cy) / max(h, 1)

    r2 = x * x + y * y
    mask = np.exp(-r2 / 0.08).astype(np.float32)
    return mask / max(float(mask.max()), 1e-6)


def fallback_edge_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=np.float32)
    mask[h // 6 : 5 * h // 6, w // 6 : 5 * w // 6] = 0.0
    return mask


def clean_background_mask(
    subject_mask: np.ndarray,
    *,
    threshold: float = 0.35,
    dilate_px: int = 17,
    center_downweight: float = 0.35,
) -> np.ndarray:
    """
    Convert subject alpha into a conservative background mask.

    We dilate the subject so hair/edges do not contaminate the background
    histogram, then lightly downweight the center because portraits often
    place the subject there.
    """
    subject = (subject_mask > threshold).astype(np.uint8)

    if dilate_px > 0:
        k = max(3, int(dilate_px))
        if k % 2 == 0:
            k += 1

        kernel = np.ones((k, k), np.uint8)
        subject = cv2.dilate(subject, kernel, iterations=1)

    bg = 1.0 - subject.astype(np.float32)

    h, w = bg.shape
    center = center_weight_mask(h, w)
    bg = bg * (1.0 - float(center_downweight) * center)

    return np.clip(bg, 0.0, 1.0)


def weighted_hist(
    values: np.ndarray,
    *,
    bins: int,
    value_range: tuple[float, float],
    weights: np.ndarray,
) -> np.ndarray:
    hist, _ = np.histogram(
        values.reshape(-1),
        bins=bins,
        range=value_range,
        weights=weights.reshape(-1),
    )

    hist = hist.astype(np.float32)
    total = float(hist.sum())

    if total > 0:
        hist /= total

    return hist


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32).reshape(-1)
    weights = weights.astype(np.float32).reshape(-1)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)

    if int(valid.sum()) < 20:
        return np.zeros(2, dtype=np.float32)

    v = values[valid]
    w = weights[valid]
    denom = max(float(w.sum()), 1e-6)

    mean = float((v * w).sum() / denom)
    std = float(np.sqrt(((v - mean) ** 2 * w).sum() / denom))

    return np.array([mean, std], dtype=np.float32)


def background_bucket_features(
    *,
    lab: np.ndarray,
    hsv: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    hue = hsv[..., 0].astype(np.float32)      # 0..179
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    lum = lab[..., 0].astype(np.float32) / 255.0

    denom = max(float(weights.sum()), 1e-6)

    def ratio(mask: np.ndarray) -> float:
        return float((weights * mask.astype(np.float32)).sum() / denom)

    warm = ((hue <= 25) | (hue >= 165)) & (sat > 0.20) & (val > 0.25)
    yellow_orange = (hue > 10) & (hue < 38) & (sat > 0.18) & (val > 0.25)
    green = (hue >= 35) & (hue <= 90) & (sat > 0.16) & (val > 0.18)
    cyan_blue = (hue >= 85) & (hue <= 130) & (sat > 0.12) & (val > 0.20)
    magenta_red = ((hue >= 130) | (hue <= 8)) & (sat > 0.18) & (val > 0.20)

    low_sat = sat < 0.12
    high_key = lum > 0.72
    dark = lum < 0.25
    saturated = sat > 0.45

    return np.array(
        [
            ratio(warm),
            ratio(yellow_orange),
            ratio(green),
            ratio(cyan_blue),
            ratio(magenta_red),
            ratio(low_sat),
            ratio(high_key),
            ratio(dark),
            ratio(saturated),
            float(weights.mean()),
        ],
        dtype=np.float32,
    )


def background_feature(rgb: np.ndarray, bg_weights: np.ndarray) -> np.ndarray:
    """
    Segmented-background feature.

    This intentionally ignores most of the subject and describes the editable
    environment:
      - background color palette
      - background tone
      - background saturation
      - rough foliage/ocean/beach/sunset color families
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    h_img, w_img = rgb.shape[:2]
    weights = bg_weights.astype(np.float32)

    if float(weights.sum()) < 100:
        weights = fallback_edge_mask(h_img, w_img)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    feats: list[np.ndarray] = [
        weighted_hist(l, bins=40, value_range=(0, 256), weights=weights),
        weighted_hist(a, bins=32, value_range=(0, 256), weights=weights),
        weighted_hist(b, bins=32, value_range=(0, 256), weights=weights),
        weighted_hist(h, bins=36, value_range=(0, 180), weights=weights),
        weighted_hist(s, bins=32, value_range=(0, 256), weights=weights),
        weighted_hist(v, bins=40, value_range=(0, 256), weights=weights),
    ]

    lf = l.astype(np.float32) / 255.0
    sf = s.astype(np.float32) / 255.0
    vf = v.astype(np.float32) / 255.0

    feats.extend([
        weighted_mean_std(lf, weights),
        weighted_mean_std(sf, weights),
        weighted_mean_std(vf, weights),
        background_bucket_features(lab=lab, hsv=hsv, weights=weights),
    ])

    out = np.concatenate(feats).astype(np.float32)
    out = l2_normalize(out.reshape(1, -1))[0]
    return out


def save_debug_images(
    *,
    debug_dir: Path,
    image_id: str,
    rgb: np.ndarray,
    subject_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    subject_mask_img = Image.fromarray(
        np.clip(subject_mask * 255, 0, 255).astype(np.uint8)
    )
    bg_mask_img = Image.fromarray(
        np.clip(bg_mask * 255, 0, 255).astype(np.uint8)
    )

    bg_rgb = rgb.copy()
    bg_rgb[bg_mask < 0.2] = 255

    subject_mask_img.save(debug_dir / f"{image_id}_subject_mask.jpg", quality=92)
    bg_mask_img.save(debug_dir / f"{image_id}_background_mask.jpg", quality=92)
    Image.fromarray(bg_rgb, mode="RGB").save(
        debug_dir / f"{image_id}_background_only.jpg",
        quality=92,
    )


def parse_debug_limit(value: int | None) -> int:
    if value is None:
        return 0

    return max(0, int(value))


def extract_background_features(
    *,
    preview_map_path: Path,
    output_npz_path: Path,
    output_metadata_path: Path,
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
    segment_model: str = DEFAULT_SEGMENT_MODEL,
    clip_weight: float = 0.50,
    background_weight: float = 0.50,
    batch_size: int = 32,
    debug_dir: Path | None = None,
    debug_limit: int = 0,
) -> BackgroundFeatureExtractResult:
    import open_clip

    if clip_weight < 0 or background_weight < 0:
        raise ValueError("clip_weight and background_weight must be >= 0")

    if clip_weight == 0 and background_weight == 0:
        raise ValueError("At least one feature weight must be greater than 0")

    _fieldnames, rows = read_csv_rows(preview_map_path)

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    device = device_for_torch()

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
    )
    clip_model.eval()
    clip_model.to(device)

    seg_model, seg_transform, seg_device = load_segmentation_model(segment_model)

    valid_rows: list[dict[str, str]] = []
    pil_images: list[Image.Image] = []
    bg_feats: list[np.ndarray] = []
    failure_rows: list[dict[str, str]] = []

    debug_written = 0
    debug_limit = parse_debug_limit(debug_limit)

    for idx, row in enumerate(rows, start=1):
        preview_path = Path(str(row.get("normalized_preview_path", "")).strip())
        image_id = str(row.get("image_id", ""))

        if not preview_path.exists():
            failure_rows.append({
                "row": str(idx),
                "image_id": image_id,
                "source_path": str(row.get("source_path", "")),
                "error": f"missing preview: {preview_path}",
            })
            continue

        try:
            pil_img = Image.open(preview_path).convert("RGB")
            rgb = np.asarray(pil_img).astype(np.uint8)

            subject_mask = predict_subject_mask(
                model=seg_model,
                transform=seg_transform,
                device=seg_device,
                image=pil_img,
            )
            bg_mask = clean_background_mask(subject_mask)

            bf = background_feature(rgb, bg_mask)

            if debug_dir is not None and debug_written < debug_limit:
                save_debug_images(
                    debug_dir=debug_dir,
                    image_id=image_id,
                    rgb=rgb,
                    subject_mask=subject_mask,
                    bg_mask=bg_mask,
                )
                debug_written += 1

            valid_rows.append(row)
            pil_images.append(pil_img)
            bg_feats.append(bf)

        except Exception as exc:
            failure_rows.append({
                "row": str(idx),
                "image_id": image_id,
                "source_path": str(row.get("source_path", "")),
                "error": repr(exc),
            })

        if idx % 50 == 0:
            print(f"processed {idx}/{len(rows)}")

    if not valid_rows:
        raise RuntimeError("No valid preview rows found. Cannot extract background features.")

    clip_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(pil_images), batch_size):
            batch_imgs = pil_images[start:start + batch_size]
            batch = torch.stack([clip_preprocess(img) for img in batch_imgs]).to(device)

            emb = clip_model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            clip_chunks.append(emb.detach().cpu().numpy().astype(np.float32))

    clip_features = np.concatenate(clip_chunks, axis=0).astype(np.float32)
    background_features = np.stack(bg_feats, axis=0).astype(np.float32)

    clip_features = l2_normalize(clip_features)
    background_features = l2_normalize(background_features)

    combined = np.concatenate(
        [
            clip_features * float(clip_weight),
            background_features * float(background_weight),
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
        background_features=background_features,
        clip_weight=np.array([clip_weight], dtype=np.float32),
        background_weight=np.array([background_weight], dtype=np.float32),
        model_name=np.array([clip_model_name]),
        pretrained=np.array([clip_pretrained]),
        segment_model=np.array([segment_model]),
        feature_version=np.array(["clip-bgseg-v1"]),
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
        failures_path = output_metadata_path.with_name(
            output_metadata_path.stem + "_failures.csv"
        )
        write_csv(
            failures_path,
            ["row", "image_id", "source_path", "error"],
            failure_rows,
        )

    return BackgroundFeatureExtractResult(
        preview_map_path=preview_map_path,
        output_npz_path=output_npz_path,
        output_metadata_path=output_metadata_path,
        total_rows=len(rows),
        extracted=len(valid_rows),
        failed=len(failure_rows),
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
        segment_model=segment_model,
        feature_dim=int(combined.shape[1]),
        clip_dim=int(clip_features.shape[1]),
        background_dim=int(background_features.shape[1]),
        clip_weight=clip_weight,
        background_weight=background_weight,
        debug_dir=debug_dir,
    )
