import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import cm
import seaborn as sns

from skimage.morphology import skeletonize
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian, threshold_otsu, threshold_local
from skimage import io as skio
from scipy import ndimage
from scipy.interpolate import griddata

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from detail import generate_visual_outputs


# ============================================================
# GLOBAL SETTINGS
# ============================================================

# Use script directory for absolute paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_SCRIPT_DIR, "SEM_figures")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 1000
FONT_SIZE = 16

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE + 2,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})


# Naming helpers
ALGORITHM_NAME_MAP = {
    "Floyd": "Floyd-Steinberg",
}


def prettify_metric_name(metric):
    mapping = {
        "porosity_%": "Porosity (%)",
        "avg_pore_area_um2": "Average Pore Area (µm²)",
        "avg_pore_equiv_diam_um": "Average Pore Diameter (µm)",
        "avg_thickness_um": "Average Thickness (µm)",
        "connectivity_index": "Connectivity Index",
        "fractal_dimension": "Fractal Dimension",
    }
    if metric in mapping:
        return mapping[metric]
    return metric.replace("_", " ").title()


# ============================================================
# 0) Helper functions
# ============================================================

def preprocess_sem(img, crop_ratio=0.88):
    h, w = img.shape
    return img[: int(h * crop_ratio), :]


def parse_power_speed(folder_name):
    """Parse power-speed from folder like '20-10' → (20, 10)."""
    m = re.match(r"^(\d+)-(\d+)$", folder_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_magnification_from_filename(filename):
    """Parse magnification from image filename suffix '..._5000x.tif'."""
    m = re.search(r"_(\d+)x\.", filename.lower())
    return int(m.group(1)) if m else None


def extract_hfw_from_image(img_rgb):
    """
    Read HFW (Horizontal Field Width) from the lower metadata panel via OCR.

    Typical metadata line: 'mag WD HFW', e.g. '5000x 7.4mm 82.9um'.
    WD is usually in mm; HFW is usually in µm (may be mm at very low mag).

    Returns:
        float or None: HFW in micrometers, or None if not found
    """
    import cv2
    try:
        import pytesseract
    except ImportError:
        return None
    
    # Bottom 12% of the image holds the metadata panel
    h = img_rgb.shape[0]
    bottom_region = img_rgb[int(h * 0.88):, :]
    
    # Convert to grayscale if RGB
    if len(bottom_region.shape) == 3:
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)
    else:
        gray = bottom_region
    
    # Boost contrast for OCR
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    try:
        text = pytesseract.image_to_string(thresh)
    except Exception:
        return None
    
    # First try µm values (HFW is usually in µm); prefer tagged HFW/HEW
    um_hfw_patterns = [
        r'HFW[:\s]*(\d+(?:\.\d+)?)\s*[uµμ]m',  # HFW 82.9um
        r'HEW[:\s]*(\d+(?:\.\d+)?)\s*[uµμ]m',  # HEW sometimes appears via OCR
    ]
    
    for pattern in um_hfw_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return float(match.group(1))
    
    # If no tag, take the largest µm value (scale bar is smaller)
    all_um_values = re.findall(r'(\d+(?:\.\d+)?)\s*[uµμ]m', text, re.IGNORECASE)
    if all_um_values:
        values = [float(v) for v in all_um_values]
        return max(values)
    
    # If no µm found, look for HFW/HEW in mm (low magnification cases)
    mm_hfw_patterns = [
        r'HFW[:\s]*(\d+(?:\.\d+)?)\s*mm',      # HFW 4.14mm
        r'HEW[:\s]*(\d+(?:\.\d+)?)\s*mm',      # HEW (OCR hatası olabilir)
    ]
    
    for pattern in mm_hfw_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return float(match.group(1)) * 1000.0  # mm → µm
    
    # Son çare: satır sonundaki mm değerini al (WD değil, HFW olabilir)
    # Format: "... WD HFW" - satırın en sonundaki değer genellikle HFW
    # Ancak WD de mm olduğundan, sadece HFW etiketi varsa mm kabul et
    
    return None


def estimate_hfw_um(magnification):
    """
    Büyütme değerinden HFW tahmin et (fallback yöntemi).
    Bu sabit (414000.0) SEM cihazına özeldir ve kalibrasyon gerektirir.
    """
    return 414000.0 / magnification


def get_micron_per_pixel(img_rgb, magnification, img_width_px):
    """
    Piksel başına mikrometre değerini hesapla.
    
    Önce OCR ile görüntüden HFW okumayı dener, başarısız olursa
    büyütme değerinden tahmin eder.
    
    Args:
        img_rgb: RGB veya grayscale görüntü (orijinal, kırpılmamış)
        magnification: Büyütme değeri (örn. 5000, 10000)
        img_width_px: Görüntü genişliği piksel cinsinden
        
    Returns:
        tuple: (micron_per_pixel, source) - source: 'ocr' veya 'estimated'
    """
    # Try OCR first
    hfw_um = extract_hfw_from_image(img_rgb)
    
    if hfw_um is not None:
        return hfw_um / img_width_px, 'ocr'
    
    # Fallback: estimate from magnification
    hfw_um = estimate_hfw_um(magnification)
    return hfw_um / img_width_px, 'estimated'


# ============================================================
# 1) Porosity
# ============================================================

def compute_porosity(img):
    blur = gaussian(img, sigma=1.0, preserve_range=True).astype(np.uint8)
    block_size = 25
    local_thresh = threshold_local(blur, block_size, method="gaussian", offset=2)
    binary = (blur <= local_thresh).astype(np.uint8) * 255
    return np.sum(binary == 255) / binary.size * 100.0


# ============================================================
# 2) Pore size
# ============================================================

def compute_pore_size(img):
    from skimage.measure import label, regionprops

    blur = gaussian(img, sigma=1.0, preserve_range=True).astype(np.uint8)
    block_size = 25
    local_thresh = threshold_local(blur, block_size, method="gaussian", offset=2)
    binary = (blur <= local_thresh).astype(np.uint8)

    labeled = label(binary, connectivity=2)
    regions = regionprops(labeled)
    areas = [r.area for r in regions]
    return np.array(areas, dtype=float)


# ============================================================
# 3) Thickness
# ============================================================

def compute_thickness(img):
    blur = gaussian(img, sigma=0.6, preserve_range=True)
    thresh = threshold_otsu(blur)
    binary = (blur > thresh).astype(np.uint8) * 255
    mask = binary == 255
    skeleton = skeletonize(mask).astype(np.uint8)
    dist = ndimage.distance_transform_edt(mask)
    thickness = dist[skeleton == 1] * 2
    return skeleton, thickness


# ============================================================
# 4) Connectivity
# ============================================================

def compute_connectivity(skeleton):
    length = np.sum(skeleton)
    if length == 0:
        return np.nan

    junctions = 0
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x]:
                n = np.sum(skeleton[y - 1:y + 2, x - 1:x + 2]) - 1
                if n >= 3:
                    junctions += 1

    return junctions / length


# ============================================================
# 5) Fractal Dimension (FIXED)
# ============================================================

def fractal_dimension(img):
    thresh = threshold_otsu(img)
    bw = img <= thresh

    min_dim = min(bw.shape)
    sizes = 2 ** np.arange(int(np.log2(min_dim)), 1, -1)

    counts = []
    for s in sizes:
        H = (bw.shape[0] // s) * s
        W = (bw.shape[1] // s) * s
        S = bw[:H, :W]
        S = S.reshape(H // s, s, W // s, s)
        boxes = S.any(axis=(1, 3))
        counts.append(boxes.sum())

    counts = np.array(counts, dtype=float)
    sizes = np.array(sizes, dtype=float)

    if len(counts) < 2:
        return np.nan

    coeff = np.polyfit(
        np.log(1.0 / sizes),
        np.log(counts + 1e-9),
        1
    )
    return coeff[0]


# ============================================================
# 6) GLCM
# ============================================================

def compute_glcm_features(img):
    img = img.astype(np.float32)
    imin, imax = img.min(), img.max()
    if imax > imin:
        img = (img - imin) / (imax - imin)
    else:
        img = np.zeros_like(img)
    img = (img * 255).astype(np.uint8)
    glcm = graycomatrix(img, [5], [0], 256, symmetric=True, normed=True)
    return {
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
    }


# ============================================================
# 7) MAIN ANALYSIS
# ============================================================

def analyze_folder(folder_path, output_csv):
    rows = []

    # folder_path
    # ├── 20-10 (power-speed)
    # │   ├── Floyd (algorithm)
    # │   ├── Jarvis
    # │   └── ...
    # └── 35-20

    for ps_name in os.listdir(folder_path):
        ps_path = os.path.join(folder_path, ps_name)
        if not os.path.isdir(ps_path):
            continue

        power, speed = parse_power_speed(ps_name)
        if power is None or speed is None:
            continue

        for alg_name in os.listdir(ps_path):
            alg_path = os.path.join(ps_path, alg_name)
            if not os.path.isdir(alg_path):
                continue

            for f in os.listdir(alg_path):
                if not f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                    continue

                full_path = os.path.join(alg_path, f)

                mag = parse_magnification_from_filename(f)
                if mag is None:
                    continue

                try:
                    # Read RGB for OCR metadata; keep grayscale for analysis
                    img_rgb = skio.imread(full_path)
                    img_raw = skio.imread(full_path, as_gray=True)
                except Exception:
                    continue

                if img_raw is None:
                    continue

                # Convert to uint8 [0, 255]
                if img_raw.dtype != np.uint8:
                    img_raw = (np.clip(img_raw, 0, 1) * 255).astype(np.uint8)

                # Piksel başına mikrometre hesapla (OCR veya tahmin)
                micron_per_pixel, scale_source = get_micron_per_pixel(
                    img_rgb, mag, img_raw.shape[1]
                )
                img = preprocess_sem(img_raw)

                fract = fractal_dimension(img)
                glcm = compute_glcm_features(img)

                porosity = pore_area_um2 = pore_diam_um = np.nan
                thick_um = conn = np.nan

                if 2500 <= mag <= 10000:
                    porosity = compute_porosity(img)
                    pores = compute_pore_size(img)
                    if len(pores):
                        pore_area_um2 = pores.mean() * micron_per_pixel ** 2
                        pore_diam_um = np.sqrt(4 * pore_area_um2 / np.pi)

                if mag >= 25000:
                    sk, th = compute_thickness(img)
                    if len(th):
                        thick_um = th.mean() * micron_per_pixel
                    conn = compute_connectivity(sk)

                rows.append({
                    "file": f,
                    "power": power,
                    "speed": speed,
                    "algorithm": alg_name,
                    "magnification": mag,
                    "micron_per_pixel": micron_per_pixel,
                    "scale_source": scale_source,
                    "porosity_%": porosity,
                    "avg_pore_area_um2": pore_area_um2,
                    "avg_pore_equiv_diam_um": pore_diam_um,
                    "avg_thickness_um": thick_um,
                    "connectivity_index": conn,
                    "fractal_dimension": fract,
                    **glcm,
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print("✔ Analysis completed →", output_csv)

    # Extra output: per (power, speed) CSV
    by_ps_dir = os.path.join(_SCRIPT_DIR, "SEM_results_by_power_speed")
    os.makedirs(by_ps_dir, exist_ok=True)

    if not df.empty and {"power", "speed"}.issubset(df.columns):
        for (p, s), sub in df.groupby(["power", "speed"]):
            fname = f"SEM_results_{p}-{s}.csv"
            sub.to_csv(os.path.join(by_ps_dir, fname), index=False)
        print("✔ Per (power,speed) CSVs saved in →", by_ps_dir)


# ============================================================
# 8) 3D BAR PLOTS
# ============================================================

def plot_3d_metric(csv_path, metric, mn, mx):
    df = pd.read_csv(csv_path)
    df = df[(df.magnification >= mn) & (df.magnification <= mx)]
    if df.empty:
        return

    algs = df.algorithm.unique()
    mags = sorted(df.magnification.unique())

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = cm.get_cmap("Reds", len(mags))

    for j, m in enumerate(mags):
        for i, a in enumerate(algs):
            v = df[(df.algorithm == a) &
                   (df.magnification == m)][metric].mean()
            ax.bar3d(i, j, 0, 0.4, 0.4, 0 if np.isnan(v) else v,
                     color=cmap(j))

    ax.set_xticks(range(len(algs)))
    xticklabels = [
        ALGORITHM_NAME_MAP.get(a, a).replace("_", " ").title()
        for a in algs
    ]
    ax.set_xticklabels(xticklabels, rotation=45, fontweight="bold")
    ax.set_yticks(range(len(mags)))
    ax.set_yticklabels([f"{m}x" for m in mags], fontweight="bold")
    ax.set_zlabel(prettify_metric_name(metric), fontweight="bold")

    # Tick etiketlerini bold yap
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.savefig(
        os.path.join(FIG_DIR, f"3Dbar_{metric}_{mn}-{mx}.png"),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 9) PCA + t-SNE
# ============================================================

def run_dimensionality_reduction(csv_path):
    df = pd.read_csv(csv_path)
    X = df.select_dtypes(float).fillna(0).values
    X = StandardScaler().fit_transform(X)

    pca = PCA(2)
    Xp = pca.fit_transform(X)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Xp[:, 0], Xp[:, 1],
                     c=df.magnification, cmap="Reds")
    cbar = plt.colorbar(sc)
    cbar.set_label("Magnification", fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    plt.xlabel("PC1", fontweight="bold")
    plt.ylabel("PC2", fontweight="bold")
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.savefig(os.path.join(FIG_DIR, "PCA.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()

    tsne = TSNE(2, perplexity=20, random_state=42)
    Xt = tsne.fit_transform(X)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Xt[:, 0], Xt[:, 1],
                     c=df.magnification, cmap="Reds")
    cbar = plt.colorbar(sc)
    cbar.set_label("Magnification", fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")

    plt.xlabel("t-SNE 1", fontweight="bold")
    plt.ylabel("t-SNE 2", fontweight="bold")
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    plt.savefig(os.path.join(FIG_DIR, "tSNE.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# 10) EXTRA VISUALIZATIONS (2D)
# ============================================================


def visualize_metrics(csv_path):
    df = pd.read_csv(csv_path)

    metrics = [
        "porosity_%",
        "avg_pore_area_um2",
        "avg_pore_equiv_diam_um",
        "avg_thickness_um",
        "connectivity_index",
        "fractal_dimension",
    ]

    def auto_ylim(values):
        """Dynamic y-limits: zoom in when spread is tiny, cap outliers when needed."""
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None

        vmin, vmax = float(arr.min()), float(arr.max())
        span = vmax - vmin

        # If all values almost equal, build a small window around them
        if span == 0 or span < 1e-6:
            pad = 0.05 * max(abs(vmax), 1.0)
            bottom = vmin - pad
            top = vmax + pad
            if bottom == top:
                top = bottom + 1.0
            return bottom, top

        # Robust caps to suppress outliers
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        p5, p95 = np.percentile(arr, [5, 95])
        cap_low = max(vmin, q1 - 1.5 * iqr, p5)
        cap_high = min(vmax, q3 + 1.5 * iqr, p95)
        cap_span = cap_high - cap_low if cap_high > cap_low else span

        pad = 0.1 * cap_span if cap_span > 0 else 0.1 * max(abs(cap_high), 1.0)
        # Decide whether to anchor to zero: if spread is small relative to level, zoom in
        small_spread = cap_span < 0.25 * max(abs(cap_high), 1e-6)
        if small_spread:
            bottom = cap_low - pad
        else:
            bottom = 0.0 if vmin >= 0 else cap_low - pad
        top = cap_high + pad

        if bottom >= top:
            top = bottom + max(1.0, abs(bottom) * 0.1)
        return bottom, top

    # 1) Boxplots per algorithm
    if "algorithm" in df.columns:
        common_mags = set()
        if "magnification" in df.columns:
            tmp = df.dropna(subset=["algorithm", "magnification"])
            if not tmp.empty:
                per_alg = tmp.groupby("algorithm")["magnification"].unique()
                if len(per_alg) > 0:
                    common_mags = set(per_alg.iloc[0])
                    for arr in per_alg.iloc[1:]:
                        common_mags &= set(arr)

        for metric in metrics:
            if metric not in df.columns:
                continue

            sub = df[["algorithm", metric]].dropna().copy()
            # Floyd yerine Floyd-Steinberg göster, diğerlerini title case yap
            sub["algorithm"] = (
                sub["algorithm"].replace(ALGORITHM_NAME_MAP)
                .astype(str)
                .str.replace("_", " ")
                .str.title()
            )
            if sub.empty:
                continue

            plt.figure(figsize=(8, 6))
            ax = sns.boxplot(data=sub, x="algorithm", y=metric, color="tab:red")
            sns.stripplot(
                data=sub,
                x="algorithm",
                y=metric,
                color="darkred",
                alpha=0.5,
                dodge=True,
            )
            display_metric = prettify_metric_name(metric)
            ax.set_xlabel("Algorithm", fontweight="bold")
            ax.set_ylabel(display_metric, fontweight="bold")

            plt.xticks(rotation=0)
            # Mean ± std above each box
            stats = (
                sub.groupby("algorithm")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            stats["std"] = stats["std"].fillna(0.0)
            xticks = ax.get_xticks()
            labels = [t.get_text() for t in ax.get_xticklabels()]
            max_y = float("-inf")
            for x, label in zip(xticks, labels):
                row = stats[stats["algorithm"] == label]
                if row.empty:
                    continue
                m = float(row["mean"].values[0])
                s = float(row["std"].values[0])
                y = m + s
                max_y = max(max_y, y)
                ax.text(
                    x,
                    y,
                    f"{m:.2f} ± {s:.2f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                        fontsize=max(FONT_SIZE - 2, 6),
                )
            # Y eksenini 0'dan başlat, robust üst sınır kullan
            ylim = auto_ylim(sub[metric])
            if ylim is not None:
                ax.set_ylim(bottom=ylim[0], top=ylim[1])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")
            plt.tight_layout()
            plt.savefig(
                os.path.join(FIG_DIR, f"box_algorithm_{metric}.png"),
                dpi=DPI,
                bbox_inches="tight",
            )
            plt.close()

            # Per-magnification boxplots for common magnifications
            if common_mags:
                for mag in sorted(common_mags):
                    sub_mag = df[
                        (df["magnification"] == mag)
                    ][["algorithm", metric]].dropna().copy()
                    if sub_mag.empty:
                        continue

                    sub_mag["algorithm"] = (
                        sub_mag["algorithm"].replace(ALGORITHM_NAME_MAP)
                        .astype(str)
                        .str.replace("_", " ")
                        .str.title()
                    )

                    plt.figure(figsize=(8, 6))
                    ax = sns.boxplot(data=sub_mag, x="algorithm", y=metric, color="tab:red")
                    sns.stripplot(
                        data=sub_mag,
                        x="algorithm",
                        y=metric,
                        color="darkred",
                        alpha=0.5,
                        dodge=True,
                    )
                    display_metric = prettify_metric_name(metric)
                    ax.set_xlabel("Algorithm", fontweight="bold")
                    ax.set_ylabel(display_metric, fontweight="bold")
                    title = f"Magnification {mag}x"
                    ax.set_title(title, fontweight="bold")

                    plt.xticks(rotation=0)
                    # Mean ± std for this magnification
                    stats = (
                        sub_mag.groupby("algorithm")[metric]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    stats["std"] = stats["std"].fillna(0.0)
                    xticks = ax.get_xticks()
                    labels = [t.get_text() for t in ax.get_xticklabels()]
                    max_y = float("-inf")
                    for x, label in zip(xticks, labels):
                        row = stats[stats["algorithm"] == label]
                        if row.empty:
                            continue
                        m = float(row["mean"].values[0])
                        s = float(row["std"].values[0])
                        y = m + s
                        max_y = max(max_y, y)
                        ax.text(
                            x,
                            y,
                            f"{m:.2f} ± {s:.2f}",
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                            fontsize=max(FONT_SIZE - 2, 6),
                        )
                    # Y eksenini 0'dan başlat, robust üst sınır kullan
                    ylim = auto_ylim(sub_mag[metric])
                    if ylim is not None:
                        ax.set_ylim(bottom=ylim[0], top=ylim[1])
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontweight("bold")
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            FIG_DIR,
                            f"box_algorithm_{metric}_mag_{mag}.png",
                        ),
                        dpi=DPI,
                        bbox_inches="tight",
                    )
                    plt.close()

                # Barplot: mean over common magnifications
                sub_common = df[
                    df["magnification"].isin(common_mags)
                ][["algorithm", metric]].dropna().copy()
                if not sub_common.empty:
                    mean_vals = (
                        sub_common.groupby("algorithm")[metric]
                        .mean()
                        .reset_index()
                    )
                    mean_vals["algorithm"] = (
                        mean_vals["algorithm"].replace(ALGORITHM_NAME_MAP)
                        .astype(str)
                        .str.replace("_", " ")
                        .str.title()
                    )

                    plt.figure(figsize=(8, 6))
                    ax = sns.barplot(
                        data=mean_vals,
                        x="algorithm",
                        y=metric,
                        color="tab:red",
                    )
                    display_metric = prettify_metric_name(metric)
                    ax.set_xlabel("Algorithm", fontweight="bold")
                    ax.set_ylabel(display_metric, fontweight="bold")
                    ax.set_title(
                        "Mean Over Common Magnifications",
                        fontweight="bold",
                    )

                    # Her barın üzerine sayısal değeri yaz
                    for patch in ax.patches:
                        height = patch.get_height()
                        ax.text(
                            patch.get_x() + patch.get_width() / 2,
                            height,
                            f"{height:.2f}",
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                            fontsize=max(FONT_SIZE - 1, 7),
                        )

                    # Y eksenini 0'dan başlat, üstte biraz boşluk bırak
                    if ax.patches:
                        max_h = max(p.get_height() for p in ax.patches)
                        ax.set_ylim(bottom=0.0, top=max_h * 1.2)

                    plt.xticks(rotation=45)
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontweight("bold")
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            FIG_DIR,
                            f"bar_algorithm_{metric}_common_mean.png",
                        ),
                        dpi=DPI,
                        bbox_inches="tight",
                    )
                    plt.close()

    # 2) Power-speed heatmaps (mean values)
    if {"power", "speed"}.issubset(df.columns):
        for metric in metrics:
            if metric not in df.columns:
                continue

            pivot = df.pivot_table(
                index="power",
                columns="speed",
                values=metric,
                aggfunc="mean",
            )
            if pivot.empty:
                continue

            plt.figure(figsize=(7, 5))
            display_metric = prettify_metric_name(metric)
            ax = sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="Reds",
                cbar_kws={"label": display_metric},
                annot_kws={"fontweight": "bold"},
            )
            ax.set_xlabel("Speed (%)", fontweight="bold")
            ax.set_ylabel("Power (%)", fontweight="bold")

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")

            # Colorbar yazılarını bold yap
            if ax.collections:
                cbar = ax.collections[0].colorbar
                if cbar is not None:
                    cbar.ax.yaxis.label.set_fontweight("bold")
                    for tick in cbar.ax.get_yticklabels():
                        tick.set_fontweight("bold")
            plt.tight_layout()
            plt.savefig(
                os.path.join(FIG_DIR, f"heatmap_power_speed_{metric}.png"),
                dpi=DPI,
                bbox_inches="tight",
            )
            plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    folder = _SCRIPT_DIR
    csv_out = os.path.join(_SCRIPT_DIR, "SEM_results_full.csv")

    # 1) Scan folders and create overlays under SEM_outputs
    generate_visual_outputs(folder)

    # 2) Then run numeric analysis for the same structure
    analyze_folder(folder, csv_out)

    for metric in [
        "porosity_%", "avg_pore_area_um2",
        "avg_thickness_um", "connectivity_index",
        "fractal_dimension"
    ]:
        plot_3d_metric(csv_out, metric, 250, 1000000)

    run_dimensionality_reduction(csv_out)

    # Extra 2D visualizations
    visualize_metrics(csv_out)
