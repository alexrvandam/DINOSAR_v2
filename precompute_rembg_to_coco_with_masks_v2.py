#!/usr/bin/env python
"""
precompute_rembg_to_coco.py

One-off preprocessing script to:
  1. Run rembg on a set of images (from a data directory and optional metadata file)
  2. Extract a foreground mask per image
  3. (Optionally) erode the mask by N pixels (v82-style)
  4. Save all masks as a single COCO JSON file (category "foreground")
  5. Save a small set of visual checks (image / mask / overlay) as PNGs
  6. **NEW**: Optionally save individual foreground masks/cutouts in multiple formats

Usage examples
--------------

# Save grayscale masks:
python precompute_rembg_to_coco.py \
  --data-dir "/path/to/images_root" \
  --output-coco "/path/to/masks.json" \
  --save-individual-masks \
  --mask-format png

# Save binary masks (pure black/white):
python precompute_rembg_to_coco.py \
  --data-dir "/path/to/images_root" \
  --output-coco "/path/to/masks.json" \
  --save-individual-masks \
  --mask-format binary

# Save actual ant images with transparent background:
python precompute_rembg_to_coco.py \
  --data-dir "/path/to/images_root" \
  --output-coco "/path/to/masks.json" \
  --save-individual-masks \
  --mask-format cutout_transparent

# Save actual ant images with black background:
python precompute_rembg_to_coco.py \
  --data-dir "/path/to/images_root" \
  --output-coco "/path/to/masks.json" \
  --save-individual-masks \
  --mask-format cutout_black

Notes
-----
- If --metadata-file is given, it must have a column "image_path"
  (relative to data-dir or absolute).
- If no metadata is given, the script walks data-dir and picks up
  *.jpg, *.jpeg, *.png, *.tif, *.tiff.
- COCO "file_name" is stored RELATIVE to data-dir. Use the same
  relative paths when you look up masks in your training script.
- Individual masks preserve the directory structure from data-dir

Format options:
  png                = Grayscale mask (0=background, 255=foreground)
  binary             = Pure binary mask (0 or 255, no gray values)
  svg                = Vector paths (scalable)
  cutout_transparent = Actual specimen with transparent background
  cutout_black       = Actual specimen with black background
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import cv2
from PIL import Image

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


# -------------------------------------------------------------------------
# Utility: v82-style mask erosion
# -------------------------------------------------------------------------

def erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    """
    Erode a binary mask by erode_px pixels using a circular kernel,
    matching the v82 behavior.

    mask: 2D array, values 0/1 or 0..255
    erode_px: number of pixels to erode (0 = no change)
    """
    if erode_px <= 0:
        return (mask > 0).astype(np.float32)

    k = max(1, int(erode_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    eroded = cv2.erode((mask * 255).astype(np.uint8), kernel)
    return (eroded > 0).astype(np.float32)


# -------------------------------------------------------------------------
# Image list loading
# -------------------------------------------------------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF")


def load_images_from_metadata(data_dir: Path, metadata_file: Path) -> List[Path]:
    """
    Read image paths from metadata. Supports CSV/TSV/JSON with an "image_path" column/key.
    Returns a list of absolute Paths.
    """
    suffix = metadata_file.suffix.lower()
    records = []

    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with metadata_file.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                if "image_path" not in row:
                    raise KeyError("Metadata file must have an 'image_path' column.")
                p = row["image_path"]
                if not p:
                    continue
                img_path = Path(p)
                if not img_path.is_absolute():
                    img_path = data_dir / img_path
                records.append(img_path)

    elif suffix == ".json":
        with metadata_file.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for row in data:
            if "image_path" not in row:
                raise KeyError("JSON metadata entries must have an 'image_path' field.")
            p = row["image_path"]
            if not p:
                continue
            img_path = Path(p)
            if not img_path.is_absolute():
                img_path = data_dir / img_path
            records.append(img_path)
    else:
        raise ValueError(f"Unsupported metadata format: {suffix}")

    # De-duplicate while preserving order
    seen = set()
    result = []
    for p in records:
        if p not in seen:
            seen.add(p)
            result.append(p)

    return result


def load_images_from_directory(data_dir: Path) -> List[Path]:
    """
    Recursively collect all image files under data_dir matching IMG_EXTS.
    """
    imgs = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                imgs.append(Path(root) / fn)
    imgs.sort()
    return imgs


# -------------------------------------------------------------------------
# Mask extraction via rembg
# -------------------------------------------------------------------------

def compute_foreground_mask(img_pil: Image.Image, mask_erode_px: int) -> np.ndarray:
    """
    Run rembg on the PIL image and return a 2D binary mask (values 0/1).
    """
    if not REMBG_AVAILABLE:
        raise RuntimeError("rembg is not installed; install with 'pip install rembg'.")

    # rembg.remove returns a PIL image with alpha channel where background is removed
    fg = rembg_remove(img_pil)
    fg_rgba = fg.convert("RGBA")
    alpha = np.array(fg_rgba.split()[-1], dtype=np.uint8)  # alpha channel

    mask = (alpha > 0).astype(np.float32)

    # Optional erosion
    mask = erode_mask(mask, mask_erode_px)

    # Handle the rare case where mask is empty
    if mask.sum() == 0:
        # Treat everything as foreground, but warn
        print("  [WARN] rembg returned empty mask; using full image as mask.")
        mask[:] = 1.0

    return mask


# -------------------------------------------------------------------------
# NEW: Individual mask saving functions
# -------------------------------------------------------------------------

def save_mask_as_png(mask: np.ndarray, output_path: Path) -> None:
    """
    Save a binary mask as a PNG file (grayscale, 0=background, 255=foreground).
    
    Args:
        mask: 2D numpy array with values 0/1 or 0..1
        output_path: Path where PNG should be saved
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_uint8, mode="L").save(output_path)


def save_mask_as_binary(mask: np.ndarray, output_path: Path) -> None:
    """
    Save a pure binary mask as PNG (0=black background, 255=white foreground, no intermediate values).
    
    Args:
        mask: 2D numpy array with values 0/1 or 0..1
        output_path: Path where binary PNG should be saved
    """
    # Convert to pure binary: 0 or 255, no gray values
    binary_mask = ((mask > 0.5).astype(np.uint8)) * 255
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary_mask, mode="L").save(output_path)


def save_foreground_cutout_transparent(
    img_pil: Image.Image,
    mask: np.ndarray,
    output_path: Path
) -> None:
    """
    Save the actual image with background removed (transparent PNG).
    This is what most people mean by "foreground mask" - the actual specimen!
    
    Args:
        img_pil: Original PIL image (RGB)
        mask: 2D binary mask array (0/1 or 0..1)
        output_path: Path where cutout PNG should be saved
    """
    # Convert to RGBA
    img_rgba = img_pil.convert("RGBA")
    img_array = np.array(img_rgba)
    
    # Apply mask to alpha channel
    mask_uint8 = (mask * 255).astype(np.uint8)
    img_array[:, :, 3] = mask_uint8  # Set alpha channel
    
    # Save as transparent PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = Image.fromarray(img_array, mode="RGBA")
    result.save(output_path, "PNG")


def save_foreground_cutout_black(
    img_pil: Image.Image,
    mask: np.ndarray,
    output_path: Path
) -> None:
    """
    Save the actual image with background set to black (RGB PNG/JPG).
    Foreground shows actual ant, background is pure black.
    
    Args:
        img_pil: Original PIL image (RGB)
        mask: 2D binary mask array (0/1 or 0..1)
        output_path: Path where cutout should be saved
    """
    img_array = np.array(img_pil)  # RGB
    
    # Expand mask to 3 channels
    mask_3d = np.stack([mask, mask, mask], axis=2)
    
    # Apply mask: foreground keeps original pixels, background becomes black
    cutout_array = (img_array * mask_3d).astype(np.uint8)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = Image.fromarray(cutout_array, mode="RGB")
    
    # Use JPEG for .jpg, PNG otherwise
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        result.save(output_path, "JPEG", quality=95)
    else:
        result.save(output_path, "PNG")


def save_mask_as_svg(
    mask: np.ndarray, 
    output_path: Path,
    image_width: int,
    image_height: int,
    simplify_epsilon: float = 1.0
) -> None:
    """
    Save a binary mask as an SVG file with path elements.
    
    Args:
        mask: 2D numpy array with values 0/1 or 0..1
        output_path: Path where SVG should be saved
        image_width: Original image width (for viewBox)
        image_height: Original image height (for viewBox)
        simplify_epsilon: Contour simplification factor (higher = simpler paths)
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        print(f"  [WARN] No contours found for SVG mask: {output_path.name}")
        # Create empty SVG
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{image_width}" 
     height="{image_height}" 
     viewBox="0 0 {image_width} {image_height}">
  <!-- Empty mask -->
</svg>'''
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.write(svg_content)
        return
    
    # Build SVG paths
    paths = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        
        # Simplify contour
        epsilon = simplify_epsilon
        simplified = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(simplified) < 3:
            continue
        
        # Build path data
        pts = simplified.reshape(-1, 2)
        path_data = f"M {pts[0][0]},{pts[0][1]}"
        
        for i in range(1, len(pts)):
            path_data += f" L {pts[i][0]},{pts[i][1]}"
        
        path_data += " Z"  # Close path
        paths.append(path_data)
    
    # Create SVG content
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{image_width}" 
     height="{image_height}" 
     viewBox="0 0 {image_width} {image_height}">
  <g id="foreground_mask" fill="white" stroke="none">
'''
    
    for i, path_data in enumerate(paths):
        svg_content += f'    <path id="mask_{i}" d="{path_data}"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(svg_content)


def save_individual_mask(
    mask: np.ndarray,
    img_pil: Optional[Image.Image],
    rel_path: Path,
    data_dir: Path,
    masks_output_dir: Path,
    mask_format: str,
    image_width: int,
    image_height: int
) -> None:
    """
    Save an individual mask file, preserving the directory structure.
    
    Args:
        mask: Binary mask array
        img_pil: Original PIL image (required for cutout formats, None for mask-only)
        rel_path: Relative path of original image from data_dir
        data_dir: Root data directory
        masks_output_dir: Output directory for masks
        mask_format: Format choice (png, binary, svg, cutout_transparent, cutout_black)
        image_width: Original image width
        image_height: Original image height
    """
    # Preserve directory structure
    rel_dir = rel_path.parent
    
    # Determine file extension and name based on format
    if mask_format == "svg":
        ext = "svg"
        suffix = "_mask"
    elif mask_format in ["cutout_transparent", "cutout_black"]:
        ext = "png"
        suffix = "_cutout"
    else:  # png, binary
        ext = "png"
        suffix = "_mask"
    
    mask_filename = f"{rel_path.stem}{suffix}.{ext}"
    output_path = masks_output_dir / rel_dir / mask_filename
    
    if mask_format == "png":
        save_mask_as_png(mask, output_path)
    elif mask_format == "binary":
        save_mask_as_binary(mask, output_path)
    elif mask_format == "svg":
        save_mask_as_svg(mask, output_path, image_width, image_height)
    elif mask_format == "cutout_transparent":
        if img_pil is None:
            raise ValueError("Image required for cutout_transparent format")
        save_foreground_cutout_transparent(img_pil, mask, output_path)
    elif mask_format == "cutout_black":
        if img_pil is None:
            raise ValueError("Image required for cutout_black format")
        save_foreground_cutout_black(img_pil, mask, output_path)
    else:
        raise ValueError(f"Unsupported mask format: {mask_format}")


# -------------------------------------------------------------------------
# COCO writing
# -------------------------------------------------------------------------

def build_coco(
    data_dir: Path,
    image_paths: List[Path],
    output_coco: Path,
    mask_erode_px: int,
    n_viz: int,
    viz_dir: Optional[Path] = None,
    save_individual_masks: bool = False,
    masks_output_dir: Optional[Path] = None,
    mask_format: str = "png",
) -> None:
    """
    Main workhorse: iterate over images, run rembg, build COCO, and save visualizations.
    
    NEW: Optionally save individual mask files for each image.
    """
    if not REMBG_AVAILABLE:
        raise RuntimeError("rembg is not installed; cannot precompute masks.")

    images = []
    annotations = []

    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    if save_individual_masks and masks_output_dir is not None:
        masks_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Individual masks will be saved to: {masks_output_dir}")
        print(f"Mask format: {mask_format.upper()}")

    ann_id = 1
    n_total = len(image_paths)
    n_viz = max(0, min(n_viz, n_total))
    viz_indices = set(range(n_viz))  # just the first N for simplicity

    for idx, abs_path in enumerate(image_paths, start=1):
        rel_path = abs_path.relative_to(data_dir)
        try:
            img_pil = Image.open(abs_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open {abs_path}: {e}")
            continue

        w, h = img_pil.size

        # Compute mask
        mask = compute_foreground_mask(img_pil, mask_erode_px=mask_erode_px)

        # ⭐ NEW: Save individual mask if requested
        if save_individual_masks and masks_output_dir is not None:
            try:
                save_individual_mask(
                    mask=mask,
                    img_pil=img_pil,  # Pass image for cutout formats
                    rel_path=rel_path,
                    data_dir=data_dir,
                    masks_output_dir=masks_output_dir,
                    mask_format=mask_format,
                    image_width=w,
                    image_height=h
                )
            except Exception as e:
                print(f"[WARN] Failed to save individual mask for {rel_path}: {e}")

        # Find contours for segmentation
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            print(f"[WARN] No contours found for {abs_path}; skipping.")
            continue

        # Build segmentation as a list of polygons (one annotation per image,
        # with potentially multiple polygons)
        segmentation: List[List[float]] = []
        all_x = []
        all_y = []

        for cnt in contours:
            if len(cnt) < 3:
                continue
            pts = cnt.reshape(-1, 2)
            xs = pts[:, 0].astype(float)
            ys = pts[:, 1].astype(float)
            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())

            poly = []
            for x, y in zip(xs, ys):
                poly.extend([float(x), float(y)])
            segmentation.append(poly)

        if not segmentation:
            print(f"[WARN] All contours degenerate for {abs_path}; skipping.")
            continue

        x_min = float(min(all_x))
        y_min = float(min(all_y))
        x_max = float(max(all_x))
        y_max = float(max(all_y))
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        area = float((mask > 0).sum())  # foreground pixel count

        image_id = idx

        images.append(
            {
                "id": image_id,
                "file_name": str(rel_path).replace(os.sep, "/"),
                "width": w,
                "height": h,
            }
        )
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,  # foreground
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
        )
        ann_id += 1

        # Optional visualization for the first n_viz images
        if viz_dir is not None and (idx - 1) in viz_indices:
            img_np = np.array(img_pil)  # RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Colorize mask
            colored = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_VIRIDIS)
            overlay = cv2.addWeighted(img_bgr, 0.6, colored, 0.4, 0)

            # Save original, mask, overlay
            base = rel_path.stem
            orig_out = viz_dir / f"{base}_image.png"
            mask_out = viz_dir / f"{base}_mask.png"
            overlay_out = viz_dir / f"{base}_overlay.png"

            Image.fromarray(img_np).save(orig_out)
            Image.fromarray(mask_uint8, mode="L").save(mask_out)
            Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(overlay_out)

            print(f"  [viz] Saved {orig_out.name}, {mask_out.name}, {overlay_out.name}")

        if idx % 50 == 0 or idx == n_total:
            print(f"Processed {idx}/{n_total} images")

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "foreground", "supercategory": "specimen"}
        ],
    }

    output_coco.parent.mkdir(parents=True, exist_ok=True)
    with output_coco.open("w") as f:
        json.dump(coco, f, indent=2)

    print(f"\n✓ COCO mask file written to: {output_coco}")
    if viz_dir is not None and n_viz > 0:
        print(f"✓ Visualizations saved to: {viz_dir}")
    if save_individual_masks and masks_output_dir is not None:
        print(f"✓ Individual masks saved to: {masks_output_dir}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute rembg-based foreground masks and save as COCO JSON."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing images",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=None,
        help="Optional metadata CSV/TSV/JSON with 'image_path' column/field",
    )
    parser.add_argument(
        "--output-coco",
        type=str,
        required=True,
        help="Path to output COCO JSON file",
    )
    parser.add_argument(
        "--mask-erode-px",
        type=int,
        default=0,
        help="Erode rembg mask by this many pixels (v82-style).",
    )
    parser.add_argument(
        "--n-viz",
        type=int,
        default=10,
        help="Number of images for which to save image/mask/overlay PNGs.",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Directory to save visual examples (default: <output_coco_dir>/viz)",
    )
    
    # NEW: Individual mask saving options
    parser.add_argument(
        "--save-individual-masks",
        action="store_true",
        help="Save individual mask files for each image (in addition to COCO JSON)",
    )
    parser.add_argument(
        "--masks-output-dir",
        type=str,
        default=None,
        help="Directory to save individual masks (default: <output_coco_dir>/masks)",
    )
    parser.add_argument(
        "--mask-format",
        type=str,
        choices=["png", "binary", "svg", "cutout_transparent", "cutout_black"],
        default="png",
        help=(
            "Format for individual masks:\n"
            "  'png' = grayscale mask (0-255)\n"
            "  'binary' = pure binary mask (0 or 255 only)\n"
            "  'svg' = vector mask paths\n"
            "  'cutout_transparent' = actual image with transparent background\n"
            "  'cutout_black' = actual image with black background"
        ),
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data-dir does not exist: {data_dir}")

    output_coco = Path(args.output_coco).expanduser().resolve()

    if args.viz_dir is not None:
        viz_dir = Path(args.viz_dir).expanduser().resolve()
    else:
        viz_dir = output_coco.parent / "viz"
    
    # NEW: Handle masks output directory
    if args.save_individual_masks:
        if args.masks_output_dir is not None:
            masks_output_dir = Path(args.masks_output_dir).expanduser().resolve()
        else:
            masks_output_dir = output_coco.parent / "masks"
    else:
        masks_output_dir = None

    if args.metadata_file:
        metadata_file = Path(args.metadata_file).expanduser().resolve()
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata-file not found: {metadata_file}")
        image_paths = load_images_from_metadata(data_dir, metadata_file)
        print(f"Loaded {len(image_paths)} images from metadata.")
    else:
        image_paths = load_images_from_directory(data_dir)
        print(f"Found {len(image_paths)} images under {data_dir}.")

    if not image_paths:
        print("No images found — nothing to do.")
        sys.exit(0)

    build_coco(
        data_dir=data_dir,
        image_paths=image_paths,
        output_coco=output_coco,
        mask_erode_px=args.mask_erode_px,
        n_viz=args.n_viz,
        viz_dir=viz_dir,
        save_individual_masks=args.save_individual_masks,
        masks_output_dir=masks_output_dir,
        mask_format=args.mask_format,
    )


if __name__ == "__main__":
    main()
