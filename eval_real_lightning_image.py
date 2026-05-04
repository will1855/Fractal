"""
Real-image sanity check for lightning fractal dimension.
Compares real lightning photographs against the generated model.

What it does:
- Processes a single image or a directory of images
- Converts to grayscale and optional crop
- Thresholds image at various levels to create binary masks
- Calculates box-counting fractal dimension D for each threshold
- Saves masks, comparative metrics, and diagnostic plots
- Compares real D values against the generated lightning model reference

Usage (single image):
  python eval_real_lightning_image.py --image path/to/lightning.jpg --out out_real_lightning

Usage (directory):
  python eval_real_lightning_image.py --image_dir data/real_lightning_crops --out out_real_lightning_all
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import lightning_dbm_model as L

# ----------------------------
# Plot helpers
# ----------------------------

def plot_box_count_rep(x, y, d_val, r2_val, out_path: str):
    """
    Mirrors the plotting logic in eval_lightning_pipeline.py
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    if len(x) >= 6:
        x_fit = x[1:-1]
        y_fit = y[1:-1]
    else:
        x_fit = x
        y_fit = y

    # Simple linear fit for visualization
    m, c = np.polyfit(x_fit, y_fit, 1)

    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-", label="Data")
    plt.plot(x_fit, m * x_fit + c, linestyle="--", label=f"Fit (D={d_val:.3f})")
    plt.xlabel("log(1/ε)")
    plt.ylabel("log N(ε)")
    plt.title(f"Box-counting: D ≈ {d_val:.3f}, R² ≈ {r2_val:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_threshold_sensitivity(thresholds, d_values, out_path: str):
    """
    Plots D versus threshold to show stability.
    """
    plt.figure()
    plt.plot(thresholds, d_values, marker="o", linestyle="-", color="tab:blue")
    plt.xlabel("Threshold")
    plt.ylabel("Fractal Dimension (D)")
    plt.title("Threshold Sensitivity: D vs Threshold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_histogram(d_values, out_path: str):
    plt.figure()
    plt.hist(d_values, bins=15, color='tab:blue', edgecolor='black', alpha=0.7)
    plt.xlabel("Fractal Dimension (D)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Fractal Dimension (Real Images)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_model_comparison(real_data, out_path: str):
    """
    real_data: list of (image_name, d_value)
    """
    plt.figure(figsize=(10, 6))
    
    names = [d[0] for d in real_data]
    vals = [d[1] for d in real_data]
    
    plt.scatter(names, vals, color='tab:blue', label='Real Images', zorder=3)
    
    # Model reference (from generated lightning metrics)
    mean_val = 1.34
    min_val = 1.30
    max_val = 1.38
    
    plt.axhline(mean_val, color='tab:red', linestyle='-', label=f'Model Mean ({mean_val})', zorder=2)
    plt.axhspan(min_val, max_val, color='tab:red', alpha=0.15, label=f'Model Range ({min_val}-{max_val})', zorder=1)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Fractal Dimension (D)")
    plt.title("Real Lightning vs Generated Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_d_by_threshold(all_results, out_path: str):
    """
    all_results: list of {image_name, threshold, D_box}
    """
    plt.figure(figsize=(10, 6))
    
    unique_images = sorted(list(set(r['image_name'] for r in all_results)))
    
    for img in unique_images:
        img_data = [r for r in all_results if r['image_name'] == img]
        img_data.sort(key=lambda x: x['threshold'])
        ths = [r['threshold'] for r in img_data]
        ds = [float(r['D_box']) for r in img_data]
        plt.plot(ths, ds, marker='o', label=img)
        
    plt.xlabel("Threshold")
    plt.ylabel("Fractal Dimension (D)")
    plt.title("Sensitivity Analysis: D by Threshold per Image")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single real lightning image")
    group.add_argument("--image_dir", type=str, help="Path to a directory of real lightning images")
    
    ap.add_argument("--out", type=str, default="out_real_lightning", help="Output directory")
    ap.add_argument("--threshold", type=int, help="Single threshold value")
    ap.add_argument("--thresholds", type=int, nargs="+", help="List of threshold values for sensitivity analysis")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"), 
                    help="Crop region: left top right bottom")
    ap.add_argument("--source_url", type=str, default="", help="Optional source URL of the image")
    
    args = ap.parse_args()

    # Determine thresholds to use
    if args.threshold is not None:
        threshold_list = [args.threshold]
        if args.thresholds:
            for t in args.thresholds:
                if t not in threshold_list:
                    threshold_list.append(t)
        primary_threshold = args.threshold
    elif args.thresholds:
        threshold_list = args.thresholds
        primary_threshold = threshold_list[0]
    else:
        threshold_list = [160, 180, 200]
        primary_threshold = 180

    out_dir = Path(args.out)
    crops_dir = out_dir / "crops"
    plots_dir = out_dir / "plots"
    
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Collect images to process
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
        source_folder = str(Path(args.image).parent)
    else:
        id_path = Path(args.image_dir)
        source_folder = str(id_path)
        
        found = []
        for ext in ['.png', '.jpg', '.jpeg']:
            found.extend(id_path.glob(f"*{ext}"))
            found.extend(id_path.glob(f"*{ext.upper()}"))
        
        # Use resolve() and set to handle duplicates on case-insensitive OS
        unique_paths = {p.resolve() for p in found}
        image_paths = sorted(list(unique_paths))

    if not image_paths:
        print("No images found to process.")
        return

    all_results = []
    
    print(f"Processing {len(image_paths)} images...")
    print(f"Thresholds: {threshold_list}")
    print("-" * 40)

    for img_path in image_paths:
        if not img_path.exists():
            print(f"Skipping: {img_path} (not found)")
            continue

        image_name = img_path.name
        image_stem = img_path.stem
        img_out_dir = crops_dir / image_stem
        os.makedirs(img_out_dir, exist_ok=True)

        try:
            img = Image.open(img_path)
            img_gray = img.convert("L")
            
            # Apply crop if provided
            crop_info = {"left": 0, "top": 0, "right": img.width, "bottom": img.height}
            if args.crop:
                left, top, right, bottom = args.crop
                img_gray = img_gray.crop((left, top, right, bottom))
                img_cropped_color = img.crop((left, top, right, bottom))
                crop_info = {"left": left, "top": top, "right": right, "bottom": bottom}
                img_cropped_color.save(img_out_dir / "cropped_original.png")
            else:
                img.save(img_out_dir / "cropped_original.png")

            gray_np = np.array(img_gray)
            height, width = gray_np.shape

            for th in threshold_list:
                mask = (gray_np >= th).astype(np.uint8)
                
                # Save mask
                mask_img = Image.fromarray(mask * 255)
                mask_rel_path = f"crops/{image_stem}/mask_th_{th}.png"
                mask_img.save(out_dir / mask_rel_path)
                
                # Calculate fractal dimension
                D, r2, bx, by = L.box_count_fractal_dimension(mask)
                
                fg_pixels = int(np.sum(mask))
                fg_fraction = fg_pixels / (width * height) if (width * height) > 0 else 0
                
                row = {
                    "image_name": image_name,
                    "source_folder": source_folder,
                    "threshold": th,
                    "crop_left": crop_info["left"],
                    "crop_top": crop_info["top"],
                    "crop_right": crop_info["right"],
                    "crop_bottom": crop_info["bottom"],
                    "width": width,
                    "height": height,
                    "foreground_pixels": fg_pixels,
                    "foreground_fraction": f"{fg_fraction:.6f}",
                    "D_box": f"{D:.6f}",
                    "box_r2": f"{r2:.6f}",
                    "mask_png": mask_rel_path
                }
                all_results.append(row)
                
                if th == primary_threshold:
                    plot_path = img_out_dir / f"box_counting_real_lightning_th_{th}.png"
                    plot_box_count_rep(bx, by, D, r2, str(plot_path))

            print(f"Completed: {image_name}")

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    # --- Finalize Metrics and Summary Plots ---

    if not all_results:
        print("No results generated.")
        return

    # 1. Combined CSV
    csv_path = out_dir / "combined_real_lightning_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)

    # 2. Summary Statistics
    d_values = [float(r['D_box']) for r in all_results]
    summary_path = out_dir / "summary_stats.csv"
    
    summary_data = {
        "metric": "D_box",
        "count": len(d_values),
        "mean": np.mean(d_values),
        "std": np.std(d_values),
        "min": np.min(d_values),
        "max": np.max(d_values)
    }
    
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_data.keys()))
        w.writeheader()
        w.writerow(summary_data)

    # 3. Summary Plots
    plot_histogram(d_values, str(plots_dir / "D_histogram.png"))
    
    # For comparison, take the primary threshold result for each image
    comp_data = []
    seen_images = set()
    for r in all_results:
        if r['image_name'] not in seen_images and r['threshold'] == primary_threshold:
            comp_data.append((r['image_name'], float(r['D_box'])))
            seen_images.add(r['image_name'])
    
    if comp_data:
        plot_model_comparison(comp_data, str(plots_dir / "D_vs_generated_model_comparison.png"))
    
    if len(threshold_list) > 1:
        plot_d_by_threshold(all_results, str(plots_dir / "D_by_image_threshold.png"))

    print("-" * 40)
    print("Done.")
    print(f"Combined CSV: {csv_path}")
    print(f"Summary Stats: {summary_path}")
    print(f"Plots:        {plots_dir}")


if __name__ == "__main__":
    main()
