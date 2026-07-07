from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class OverrideTemplateResult:
    input_path: Path
    output_path: Path
    candidate_count: int


@dataclass(frozen=True)
class ContactSheetResult:
    preview_map_path: Path
    output_dir: Path
    sheets_written: int
    image_count: int


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


def split_candidate_keys(value: str) -> list[str]:
    out: list[str] = []

    for item in str(value or "").split(";"):
        key = item.strip()
        if key:
            out.append(key)

    return out


def row_candidate_key(row: dict[str, str]) -> str:
    primary = str(row.get("primary_candidate_key", "")).strip()
    if primary:
        return primary

    keys = split_candidate_keys(str(row.get("candidate_keys", "")))
    if keys:
        return keys[0]

    return ""


def make_overrides_template(
    *,
    input_path: Path,
    output_path: Path,
    include_examples: bool = False,
) -> OverrideTemplateResult:
    _fieldnames, rows = read_csv_rows(input_path)

    candidate_keys = sorted({
        row_candidate_key(row)
        for row in rows
        if row_candidate_key(row)
    })

    out_rows: list[dict[str, str]] = []

    for key in candidate_keys:
        out_rows.append({
            "candidate_key": key,
            "wb": "",
            "exposure_mode": "",
            "target_median": "",
            "low_pct": "",
            "high_pct": "",
            "user_wb": "",
            "notes": "",
        })

    if include_examples:
        out_rows.insert(0, {
            "candidate_key": "example_custom",
            "wb": "custom",
            "exposure_mode": "center-midtone",
            "target_median": "0.38",
            "low_pct": "",
            "high_pct": "",
            "user_wb": "2.0,1.0,1.45,1.0",
            "notes": "example only; delete this row before real use",
        })
        out_rows.insert(0, {
            "candidate_key": "example_percentile",
            "wb": "daylight",
            "exposure_mode": "percentile",
            "target_median": "",
            "low_pct": "0.5",
            "high_pct": "99.5",
            "user_wb": "",
            "notes": "example only; delete this row before real use",
        })

    fields = [
        "candidate_key",
        "wb",
        "exposure_mode",
        "target_median",
        "low_pct",
        "high_pct",
        "user_wb",
        "notes",
    ]

    write_csv(output_path, fields, out_rows)

    return OverrideTemplateResult(
        input_path=input_path,
        output_path=output_path,
        candidate_count=len(candidate_keys),
    )


def default_font(size: int = 16) -> ImageFont.ImageFont:
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


def fit_image(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size

    if w <= 0 or h <= 0:
        return Image.new("RGB", (box_w, box_h), "white")

    scale = min(box_w / w, box_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (box_w, box_h), "white")
    x = (box_w - new_w) // 2
    y = (box_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def text_lines_for_row(row: dict[str, str]) -> list[str]:
    settings = (
        f"{row.get('wb_mode', '')} | "
        f"{row.get('exposure_mode', '')} | "
        f"tm={row.get('target_median', '')}"
    )

    override = row.get("override_applied", "")
    if override == "true":
        settings += " | override"

    return [
        str(row.get("file_name", "")),
        settings,
        str(row.get("source_path", "")),
    ]


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    max_lines: int,
    line_height: int,
) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        test = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)

        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

        if len(lines) >= max_lines:
            break

    if current and len(lines) < max_lines:
        lines.append(current)

    for idx, line in enumerate(lines[:max_lines]):
        if idx == max_lines - 1 and len(lines) > max_lines:
            line = line.rstrip() + "..."
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height

    return y


def make_contact_sheet_for_group(
    *,
    group_key: str,
    rows: list[dict[str, str]],
    output_path: Path,
    thumb_size: int,
    columns: int,
) -> None:
    title_h = 46
    label_h = 92
    pad = 16
    gap = 12

    font_title = default_font(22)
    font_label = default_font(13)

    rows_count = max(1, math.ceil(len(rows) / columns))

    cell_w = thumb_size
    cell_h = thumb_size + label_h

    canvas_w = pad * 2 + columns * cell_w + (columns - 1) * gap
    canvas_h = pad * 2 + title_h + rows_count * cell_h + (rows_count - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"{group_key}   ({len(rows)} images)"
    draw.text((pad, pad), title, font=font_title, fill="black")

    for idx, row in enumerate(rows):
        col = idx % columns
        r = idx // columns

        x = pad + col * (cell_w + gap)
        y = pad + title_h + r * (cell_h + gap)

        preview_path = Path(str(row.get("normalized_preview_path", "")))

        try:
            img = Image.open(preview_path)
            thumb = fit_image(img, thumb_size, thumb_size)
        except Exception:
            thumb = Image.new("RGB", (thumb_size, thumb_size), "lightgray")
            d = ImageDraw.Draw(thumb)
            d.text((12, 12), "missing preview", font=font_label, fill="black")

        canvas.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumb_size, y + thumb_size), outline="black", width=1)

        ty = y + thumb_size + 8
        lines = text_lines_for_row(row)

        draw.text((x, ty), lines[0], font=font_label, fill="black")
        ty += 18

        draw.text((x, ty), lines[1], font=font_label, fill="black")
        ty += 18

        draw_wrapped_text(
            draw,
            lines[2],
            (x, ty),
            font_label,
            "gray",
            max_width=thumb_size,
            max_lines=3,
            line_height=15,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "JPEG", quality=92, optimize=True)


def make_contact_sheets(
    *,
    preview_map_path: Path,
    output_dir: Path,
    thumb_size: int = 320,
    columns: int = 3,
) -> ContactSheetResult:
    _fieldnames, rows = read_csv_rows(preview_map_path)

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        key = row_candidate_key(row) or "unknown"
        by_group[key].append(row)

    sheets_written = 0

    for group_key in sorted(by_group.keys()):
        group_rows = sorted(
            by_group[group_key],
            key=lambda r: (
                str(r.get("capture_time", "")),
                str(r.get("file_name", "")),
                str(r.get("source_path", "")),
            ),
        )

        output_path = output_dir / f"{group_key}.jpg"

        make_contact_sheet_for_group(
            group_key=group_key,
            rows=group_rows,
            output_path=output_path,
            thumb_size=thumb_size,
            columns=columns,
        )

        sheets_written += 1

    return ContactSheetResult(
        preview_map_path=preview_map_path,
        output_dir=output_dir,
        sheets_written=sheets_written,
        image_count=len(rows),
    )
