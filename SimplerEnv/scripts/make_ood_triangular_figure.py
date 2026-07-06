#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_IND = "SimplerEnv/scripts/ood_showcase/01_ind_base_main_train/frame0.png"
DEFAULT_OOD1 = "SimplerEnv/scripts/ood_showcase/02_ood_visual_texture/frame0.png"
DEFAULT_OOD2 = "SimplerEnv/scripts/ood_showcase/03_ood_multi_receptacle/frame0.png"

DEFAULT_TEXT_IND = "IND Scene"
DEFAULT_TEXT_OOD1 = "OOD Texture (cobblestone)"
DEFAULT_TEXT_OOD2 = "OOD Multiple receptacles (newspaper + frying pan)"


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try common system fonts first; fall back to PIL default.
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_label(
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    margin: int = 14,
    pad_x: int = 12,
    pad_y: int = 8,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    x0 = margin
    y1 = image.height - margin
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y
    y0 = y1 - box_h
    x1 = x0 + box_w
    # Semi-opaque dark rounded rectangle.
    draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=(0, 0, 0, 170))
    draw.text((x0 + pad_x, y0 + pad_y - 1), text, font=font, fill=(255, 255, 255, 255))


def _prepare_panel(path: Path, label: str, panel_w: int, panel_h: int, font: ImageFont.ImageFont) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    img = Image.open(path).convert("RGB")
    # Crop to fill target ratio, then resize.
    src_w, src_h = img.size
    target_ratio = panel_w / panel_h
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    img = img.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    _draw_label(img, label, font=font)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose a triangular IND/OOD figure: IND on top, two OOD panels on bottom, "
            "with rounded bottom-left text labels."
        )
    )
    parser.add_argument("--ind", default=DEFAULT_IND, help="Path to IND image")
    parser.add_argument("--ood1", default=DEFAULT_OOD1, help="Path to OOD image (left bottom)")
    parser.add_argument("--ood2", default=DEFAULT_OOD2, help="Path to OOD image (right bottom)")
    parser.add_argument("--text-ind", default=DEFAULT_TEXT_IND)
    parser.add_argument("--text-ood1", default=DEFAULT_TEXT_OOD1)
    parser.add_argument("--text-ood2", default=DEFAULT_TEXT_OOD2)
    parser.add_argument("--out", default="SimplerEnv/scripts/ood_showcase/ood_triangular_layout.png")
    parser.add_argument("--panel-width", type=int, default=720)
    parser.add_argument("--panel-height", type=int, default=430)
    parser.add_argument("--gap", type=int, default=28, help="Gap between panels")
    parser.add_argument("--canvas-pad", type=int, default=30)
    parser.add_argument("--font-size", type=int, default=30)
    args = parser.parse_args()

    panel_w = args.panel_width
    panel_h = args.panel_height
    gap = args.gap
    pad = args.canvas_pad
    font = _load_font(args.font_size)

    ind_img = _prepare_panel(Path(args.ind), args.text_ind, panel_w, panel_h, font)
    ood1_img = _prepare_panel(Path(args.ood1), args.text_ood1, panel_w, panel_h, font)
    ood2_img = _prepare_panel(Path(args.ood2), args.text_ood2, panel_w, panel_h, font)

    # Triangular layout:
    # top row: one centered panel
    # bottom row: two panels
    canvas_w = 2 * panel_w + gap + 2 * pad
    canvas_h = 2 * panel_h + gap + 2 * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(246, 246, 246))

    top_x = (canvas_w - panel_w) // 2
    top_y = pad
    bottom_y = pad + panel_h + gap
    left_x = pad
    right_x = pad + panel_w + gap

    canvas.paste(ind_img, (top_x, top_y))
    canvas.paste(ood1_img, (left_x, bottom_y))
    canvas.paste(ood2_img, (right_x, bottom_y))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
