import os
import re
import cv2
import numpy as np
from skimage.morphology import skeletonize


# ============================================================
# 0) Output folder
# ============================================================

# Use script directory for absolute paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "SEM_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1) Parsing helpers
# ============================================================

def parse_magnification(folder_name):
    folder_name = folder_name.lower()
    if folder_name.endswith("x") and folder_name[:-1].isdigit():
        return int(folder_name[:-1])
    return None


def parse_magnification_from_filename(filename):
    """Read magnification from filename: '..._5000x.tif' → 5000."""
    m = re.search(r"_(\d+)x\.", filename.lower())
    return int(m.group(1)) if m else None


def parse_power_speed_from_path(root, base_folder):
    """Find power-speed like '20-10' from the path relative to base."""
    rel = os.path.relpath(root, base_folder)
    parts = rel.split(os.sep)
    for part in parts:
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def parse_algorithm_from_path(root, base_folder):
    """Grab algorithm name (Floyd, Jarvis, Standard, Stucki) from path parts."""
    rel = os.path.relpath(root, base_folder)
    parts = rel.split(os.sep)
    alg_set = {"Floyd", "Jarvis", "Standard", "Stucki"}
    for part in parts:
        if part in alg_set:
            return part
    return None


# ============================================================
# 2) Crop SEM panel
# ============================================================

def preprocess_sem(img, crop_ratio=0.88):
    h, w = img.shape
    return img[: int(h * crop_ratio), :]


# ============================================================
# 3) Pore segmentation (adaptive threshold)
# ============================================================

def get_pore_binary(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 2
    )
    return binary


# ============================================================
# 4) Save pore overlay
# ============================================================

def save_pore_overlay(img, pore_binary, save_path, meta_text=None):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = img_color.copy()

    # Pores in red
    overlay[pore_binary == 255] = (0, 0, 255)

    blended = cv2.addWeighted(overlay, 0.45, img_color, 0.55, 0)

    # Write meta text at bottom right
    if meta_text:
        h, w, _ = blended.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        margin = 5

        (text_w, text_h), _ = cv2.getTextSize(meta_text, font, font_scale, thickness)
        x = max(margin, w - text_w - margin)
        y = max(text_h + margin, h - margin)

        # Background rectangle for readability
        cv2.rectangle(
            blended,
            (x - 2, y - text_h - 4),
            (x + text_w + 2, y + 2),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(blended, meta_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(save_path, blended)


# ============================================================
# 5) Thickness + skeleton extraction
# ============================================================

def compute_skeleton(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = binary == 255
    skeleton = skeletonize(mask).astype(np.uint8)

    return skeleton


# ============================================================
# 6) Extract connectivity points
# ============================================================

def compute_connectivity_points(skeleton):
    h, w = skeleton.shape
    points = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 1:
                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1
                if neighbors >= 3:
                    points.append((y, x))

    return points


# ============================================================
# 7) Save skeleton + connectivity overlay
# ============================================================

def save_skeleton_overlay(img, skeleton, conn_points, save_path, meta_text=None):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Skeleton in white
    img_color[skeleton == 1] = (255, 255, 255)

    # Junction points in blue
    for (y, x) in conn_points:
        cv2.circle(img_color, (x, y), 3, (255, 0, 0), -1)

    # Write meta text at bottom right
    if meta_text:
        h, w, _ = img_color.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        margin = 5

        (text_w, text_h), _ = cv2.getTextSize(meta_text, font, font_scale, thickness)
        x = max(margin, w - text_w - margin)
        y = max(text_h + margin, h - margin)

        cv2.rectangle(
            img_color,
            (x - 2, y - text_h - 4),
            (x + text_w + 2, y + 2),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(img_color, meta_text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(save_path, img_color)


# ============================================================
# 8) MAIN EXECUTION
# ============================================================

def generate_visual_outputs(folder):
    print(f"\n📁 Scanning folder: {folder}")

    for root, dirs, files in os.walk(folder):
        # Skip output, cache, git, and SEM_ prefixed auxiliary folders
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and not d.startswith("SEM_") and d != "__pycache__"
        ]

        folder_name = os.path.basename(root)
        mag = parse_magnification(folder_name)
        print(f"\n🔍 Scanning subfolder: {root}")

        # Extract power-speed from path (e.g., 20-10)
        power, speed = parse_power_speed_from_path(root, folder)

        # Extract algorithm name from path
        alg_name = parse_algorithm_from_path(root, folder)

        # Optional power-speed tag for filenames
        ps_tag = None
        if power is not None and speed is not None:
            ps_tag = f"{power}-{speed}"

        # Mirror input folder structure under SEM_outputs
        rel_dir = os.path.relpath(root, folder)
        out_dir = os.path.join(OUTPUT_DIR, rel_dir) if rel_dir != "." else OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                continue

            full_path = os.path.join(root, file)
            print(f"Processing: {full_path}")

            img_raw = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                print("⚠️ Error reading image")
                continue

            img = preprocess_sem(img_raw)

            base = os.path.splitext(file)[0]

            # Determine magnification (folder name first, else filename)
            img_mag = mag
            if img_mag is None:
                img_mag = parse_magnification_from_filename(file)

            # Skip if magnification is missing
            if img_mag is None:
                print("⚠️ Skipping image (magnification not found):", full_path)
                continue

            # Filename suffix: original + optional power-speed tag
            name_suffix = base
            if ps_tag is not None:
                name_suffix = f"{base}_{ps_tag}"

            # Compose meta info text for overlays
            meta_parts = []
            if power is not None and speed is not None:
                meta_parts.append(f"P={power}%, S={speed}%")
            if alg_name is not None:
                meta_parts.append(f"Alg={alg_name}")
            if img_mag is not None:
                meta_parts.append(f"Mag={img_mag}x")
            meta_text = " | ".join(meta_parts) if meta_parts else None

            # Medium magnification → pore overlay
            if 2500 <= img_mag <= 10000:
                pore_binary = get_pore_binary(img)
                save_path = os.path.join(out_dir, f"{name_suffix}_pore_overlay.png")
                save_pore_overlay(img, pore_binary, save_path, meta_text=meta_text)
                print(f"   ✔ Saved pore overlay → {save_path}")

            # High magnification → skeleton + connectivity
            if img_mag >= 25000:
                skeleton = compute_skeleton(img)
                conn_points = compute_connectivity_points(skeleton)
                save_path = os.path.join(out_dir, f"{name_suffix}_connectivity_overlay.png")
                save_skeleton_overlay(img, skeleton, conn_points, save_path, meta_text=meta_text)
                print(f"   ✔ Saved skeleton/connectivity overlay → {save_path}")


# ============================================================
# 9) MAIN
# ============================================================

if __name__ == "__main__":
    folder = r"C:\Users\saatz\Desktop\Projects\SEM-Analysis\SEM_images"  # değiştirilebilir
    generate_visual_outputs(folder)
    print("\n🎉 ALL VISUAL OUTPUTS SAVED IN → SEM_outputs/")
