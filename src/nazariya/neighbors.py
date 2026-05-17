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


def feature_label(data: np.lib.npyio.NpzFile) -> str:
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

    if clip_weight and background_weight:
        label = f"CLIP {clip_weight} / bg {background_weight}"
        if version:
            label = f"{label} | {version}"
        return label

    if clip_weight and color_weight:
        return f"CLIP {clip_weight} / color {color_weight}"

    return "features"


def generate_neighbor_sheets(
    *,
    features_path: Path,
    preview_map_path: Path,
    output_dir: Path,
    top_k: int = 10,
    exclude_same_candidate: bool = True,
    thumb_size: int = 260,
    label_height: int = 92,
) -> NeighborSheetResult:
    if top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    data = np.load(features_path)

    features = data["features"]
    image_ids = [str(x) for x in data["image_ids"]]
    candidates = [str(x) for x in data["candidate_keys"]]
    preview_paths = [str(x) for x in data["preview_paths"]]

    if len(features) != len(image_ids):
        raise ValueError("features and image_ids length mismatch")

    preview_map = read_preview_map(preview_map_path)

    sim = features @ features.T
    np.fill_diagonal(sim, -1)

    by_candidate: dict[str, list[int]] = defaultdict(list)

    for idx, candidate in enumerate(candidates):
        by_candidate[candidate].append(idx)

    output_dir.mkdir(parents=True, exist_ok=True)

    pad = 18
    gap = 12
    cols = top_k + 1
    cell_h = thumb_size + label_height
    canvas_w = pad * 2 + cols * thumb_size + (cols - 1) * gap
    feature_text = feature_label(data)

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
    )
