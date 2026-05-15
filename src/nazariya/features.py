from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


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


def color_light_features(rgb: np.ndarray) -> np.ndarray:
    """
    Hand-built color/light feature.

    This is intentionally simple and stable:
    - Lab L histogram for tone
    - Lab a/b histograms for color opponent axes
    - HSV saturation histogram
    - HSV value histogram
    - simple contrast stats

    Output is L2-normalized.
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    s = hsv[..., 1]
    v = hsv[..., 2]

    feats: list[np.ndarray] = []

    for channel, bins, value_range in [
        (l, 32, (0, 256)),
        (a, 24, (0, 256)),
        (b, 24, (0, 256)),
        (s, 24, (0, 256)),
        (v, 32, (0, 256)),
    ]:
        hist, _ = np.histogram(channel, bins=bins, range=value_range)
        hist = hist.astype(np.float32)
        hist = hist / max(hist.sum(), 1.0)
        feats.append(hist)

    # Tone/contrast summary.
    vf = v.astype(np.float32) / 255.0
    lf = l.astype(np.float32) / 255.0
    sf = s.astype(np.float32) / 255.0

    stats = np.array(
        [
            float(np.mean(lf)),
            float(np.std(lf)),
            float(np.percentile(lf, 5)),
            float(np.percentile(lf, 50)),
            float(np.percentile(lf, 95)),
            float(np.mean(sf)),
            float(np.std(sf)),
            float(np.mean(vf)),
            float(np.std(vf)),
        ],
        dtype=np.float32,
    )
    feats.append(stats)

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

    # Weighted concatenation, then normalize again.
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
    )
