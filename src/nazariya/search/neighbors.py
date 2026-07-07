from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class NeighborSheetResult:
    features_path: Path
    preview_map_path: Path
    output_dir: Path
    sheets_written: int
    image_count: int
    candidate_count: int
    top_k: int
    exclude_same_candidate: bool
    feature_space: str
    metric: str
    clip_pool: int


VALID_FEATURE_SPACES = {
    "combined",
    "clip",
    "color",
    "background",
    "clip-then-background",
}

VALID_METRICS = {
    "cosine",
    "histogram-intersection",
    "bhattacharyya",
}


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        return list(reader.fieldnames), list(reader)


def default_font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]

    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except Exception:
            pass

    return ImageFont.load_default()


FONT = default_font(14)
FONT_TITLE = default_font(22)


def short_text(text: str, n: int = 42) -> str:
    text = str(text or "")

    if len(text) <= n:
        return text

    return text[: n - 3] + "..."


def read_preview_map(path: Path) -> dict[str, dict[str, str]]:
    _fieldnames, rows = read_csv_rows(path)
    return {str(r.get("image_id", "")): r for r in rows if str(r.get("image_id", ""))}


def fit_image(path: Path, size: int) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), "lightgray")
        d = ImageDraw.Draw(img)
        d.text((12, 12), "missing", font=FONT, fill="black")
        return img

    w, h = img.size

    if w <= 0 or h <= 0:
        return Image.new("RGB", (size, size), "white")

    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (size, size), "white")
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, lines: list[str]) -> None:
    yy = y

    for line in lines:
        draw.text((x, yy), short_text(line), font=FONT, fill="black")
        yy += 17


def row_label(
    *,
    meta: dict[str, str],
    image_id: str,
    candidate: str,
    prefix: str,
) -> list[str]:
    file_name = meta.get("file_name", "")

    return [
        prefix,
        f"{candidate} | {file_name}",
        f"id {image_id}",
    ]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def as_distribution(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    x = np.maximum(x, 0.0)
    row_sum = x.sum(axis=1, keepdims=True)
    return x / np.maximum(row_sum, eps)


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    x = l2_normalize(features.astype(np.float32))
    return x @ x.T


def histogram_intersection_similarity_matrix(features: np.ndarray) -> np.ndarray:
    x = as_distribution(features)
    n = x.shape[0]
    sim = np.empty((n, n), dtype=np.float32)

    for i in range(n):
        sim[i] = np.minimum(x[i][None, :], x).sum(axis=1)

    return sim


def bhattacharyya_similarity_matrix(features: np.ndarray) -> np.ndarray:
    x = as_distribution(features)
    sqrt_x = np.sqrt(x)
    return sqrt_x @ sqrt_x.T


def metric_similarity_matrix(features: np.ndarray, metric: str) -> np.ndarray:
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Invalid metric: {metric}. Valid: {sorted(VALID_METRICS)}"
        )

    if metric == "cosine":
        return cosine_similarity_matrix(features)

    if metric == "histogram-intersection":
        return histogram_intersection_similarity_matrix(features)

    if metric == "bhattacharyya":
        return bhattacharyya_similarity_matrix(features)

    raise ValueError(f"Unhandled metric: {metric}")


def choose_feature_matrix(data: np.lib.npyio.NpzFile, feature_space: str) -> np.ndarray:
    if feature_space == "combined":
        return np.asarray(data["features"], dtype=np.float32)

    if feature_space == "clip":
        if "clip_features" not in data.files:
            raise ValueError("Feature file does not contain clip_features")
        return np.asarray(data["clip_features"], dtype=np.float32)

    if feature_space == "color":
        if "color_features" not in data.files:
            raise ValueError("Feature file does not contain color_features")
        return np.asarray(data["color_features"], dtype=np.float32)

    if feature_space == "background":
        if "background_features" not in data.files:
            raise ValueError("Feature file does not contain background_features")
        return np.asarray(data["background_features"], dtype=np.float32)

    raise ValueError(f"Unhandled feature_space: {feature_space}")


def feature_label(
    data: np.lib.npyio.NpzFile,
    feature_space: str,
    metric: str,
    clip_pool: int,
) -> str:
    clip_weight = ""
    color_weight = ""
    background_weight = ""
    version = ""

    if "clip_weight" in data.files:
        try:
            clip_weight = f"{float(data['clip_weight'][0]):.2f}"
        except Exception:
            clip_weight = ""

    if "color_weight" in data.files:
        try:
            color_weight = f"{float(data['color_weight'][0]):.2f}"
        except Exception:
            color_weight = ""

    if "background_weight" in data.files:
        try:
            background_weight = f"{float(data['background_weight'][0]):.2f}"
        except Exception:
            background_weight = ""

    if "feature_version" in data.files:
        try:
            version = str(data["feature_version"][0])
        except Exception:
            version = ""

    if "color_feature_version" in data.files:
        try:
            version = str(data["color_feature_version"][0])
        except Exception:
            pass

    parts = [f"space={feature_space}", f"metric={metric}"]

    if feature_space == "clip-then-background":
        parts.append(f"clip_pool={clip_pool}")

    if clip_weight and background_weight:
        parts.append(f"CLIP {clip_weight} / bg {background_weight}")
    elif clip_weight and color_weight:
        parts.append(f"CLIP {clip_weight} / color {color_weight}")

    if version:
        parts.append(version)

    return " | ".join(parts)


def two_stage_clip_then_background_similarity(
    *,
    clip_features: np.ndarray,
    background_features: np.ndarray,
    background_metric: str,
    clip_pool: int,
) -> np.ndarray:
    """
    Stage 1:
      For each query, get top N candidates by CLIP cosine similarity.

    Stage 2:
      Within that CLIP pool only, score/rerank by segmented-background histogram.

    Output:
      A sparse-like similarity matrix where non-pool items are -1.
      Higher is better.
    """
    if clip_pool <= 0:
        raise ValueError("--clip-pool must be greater than zero for clip-then-background")

    clip_sim = cosine_similarity_matrix(clip_features)
    bg_sim = metric_similarity_matrix(background_features, metric=background_metric)

    n = clip_sim.shape[0]
    out = np.full((n, n), -1.0, dtype=np.float32)

    for i in range(n):
        clip_scores = clip_sim[i].copy()
        clip_scores[i] = -1.0

        pool_size = min(clip_pool, n - 1)
        pool = np.argsort(-clip_scores)[:pool_size]

        out[i, pool] = bg_sim[i, pool]

    return out


def build_similarity_matrix(
    data: np.lib.npyio.NpzFile,
    *,
    feature_space: str,
    metric: str,
    clip_pool: int,
) -> np.ndarray:
    if feature_space not in VALID_FEATURE_SPACES:
        raise ValueError(
            f"Invalid feature_space: {feature_space}. "
            f"Valid: {sorted(VALID_FEATURE_SPACES)}"
        )

    if feature_space == "clip-then-background":
        if "clip_features" not in data.files:
            raise ValueError("Feature file does not contain clip_features")

        if "background_features" not in data.files:
            raise ValueError("Feature file does not contain background_features")

        clip_features = np.asarray(data["clip_features"], dtype=np.float32)
        background_features = np.asarray(data["background_features"], dtype=np.float32)

        if metric == "cosine":
            # For this mode, cosine means:
            #   Stage 1 CLIP cosine, Stage 2 background cosine.
            background_metric = "cosine"
        elif metric in {"histogram-intersection", "bhattacharyya"}:
            background_metric = metric
        else:
            raise ValueError(
                "clip-then-background supports metric: cosine, "
                "histogram-intersection, bhattacharyya"
            )

        return two_stage_clip_then_background_similarity(
            clip_features=clip_features,
            background_features=background_features,
            background_metric=background_metric,
            clip_pool=clip_pool,
        )

    features = choose_feature_matrix(data, feature_space)
    return metric_similarity_matrix(features, metric=metric)


def generate_neighbor_sheets(
    *,
    features_path: Path,
    preview_map_path: Path,
    output_dir: Path,
    top_k: int = 10,
    exclude_same_candidate: bool = True,
    thumb_size: int = 260,
    label_height: int = 92,
    feature_space: str = "combined",
    metric: str = "cosine",
    clip_pool: int = 80,
) -> NeighborSheetResult:
    if top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    data = np.load(features_path)

    image_ids = [str(x) for x in data["image_ids"]]
    candidates = [str(x) for x in data["candidate_keys"]]
    preview_paths = [str(x) for x in data["preview_paths"]]

    sim = build_similarity_matrix(
        data,
        feature_space=feature_space,
        metric=metric,
        clip_pool=clip_pool,
    )

    if sim.shape[0] != len(image_ids):
        raise ValueError("similarity matrix and image_ids length mismatch")

    np.fill_diagonal(sim, -1)

    preview_map = read_preview_map(preview_map_path)

    by_candidate: dict[str, list[int]] = defaultdict(list)

    for idx, candidate in enumerate(candidates):
        by_candidate[candidate].append(idx)

    output_dir.mkdir(parents=True, exist_ok=True)

    pad = 18
    gap = 12
    cols = top_k + 1
    cell_h = thumb_size + label_height
    canvas_w = pad * 2 + cols * thumb_size + (cols - 1) * gap

    feature_text = feature_label(
        data,
        feature_space=feature_space,
        metric=metric,
        clip_pool=clip_pool,
    )

    sheets_written = 0

    for candidate_key in sorted(by_candidate.keys()):
        query_indices = by_candidate[candidate_key]
        rows_count = len(query_indices)

        canvas_h = pad * 2 + 44 + rows_count * cell_h + (rows_count - 1) * gap
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)

        title = f"{candidate_key} cross-set nearest neighbors | {feature_text}"

        if not exclude_same_candidate:
            title = f"{candidate_key} nearest neighbors | {feature_text}"

        draw.text(
            (pad, pad),
            title,
            font=FONT_TITLE,
            fill="black",
        )

        for row_i, query_idx in enumerate(query_indices):
            scores = sim[query_idx].copy()

            if exclude_same_candidate:
                for j, cand in enumerate(candidates):
                    if cand == candidate_key:
                        scores[j] = -1

            order = np.argsort(-scores)[:top_k]
            indices = [query_idx] + list(order)

            for col_i, idx in enumerate(indices):
                x = pad + col_i * (thumb_size + gap)
                y = pad + 44 + row_i * (cell_h + gap)

                img = fit_image(Path(preview_paths[idx]), thumb_size)
                canvas.paste(img, (x, y))

                outline = "red" if col_i == 0 else "black"
                width = 4 if col_i == 0 else 1
                draw.rectangle(
                    (x, y, x + thumb_size, y + thumb_size),
                    outline=outline,
                    width=width,
                )

                meta = preview_map.get(image_ids[idx], {})

                if col_i == 0:
                    label = row_label(
                        meta=meta,
                        image_id=image_ids[idx],
                        candidate=candidates[idx],
                        prefix=f"QUERY #{query_idx}",
                    )
                else:
                    label = row_label(
                        meta=meta,
                        image_id=image_ids[idx],
                        candidate=candidates[idx],
                        prefix=f"rank {col_i} | sim {scores[idx]:.3f}",
                    )

                draw_label(draw, x, y + thumb_size + 8, label)

        out_path = output_dir / f"{candidate_key}.jpg"
        canvas.save(out_path, "JPEG", quality=92, optimize=True)
        sheets_written += 1

    return NeighborSheetResult(
        features_path=features_path,
        preview_map_path=preview_map_path,
        output_dir=output_dir,
        sheets_written=sheets_written,
        image_count=len(image_ids),
        candidate_count=len(by_candidate),
        top_k=top_k,
        exclude_same_candidate=exclude_same_candidate,
        feature_space=feature_space,
        metric=metric,
        clip_pool=clip_pool,
    )
