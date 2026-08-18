import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
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

DPI = 300
FONT_SIZE = 12
FONT_FAMILY = "Times New Roman"

def setup_publication_style():
    """Configure matplotlib for publication-quality figures with Times New Roman."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.titlesize": FONT_SIZE,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "mathtext.fontset": "stix",  # Math font compatible with Times
        "figure.dpi": 150,
        "savefig.dpi": DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,  # TrueType fonts in PDF
        "ps.fonttype": 42,
    })

# Apply publication style at module load
setup_publication_style()


# ============================================================
# PUBLICATION COLOR SCHEME
# ============================================================

# Professional grayscale/muted palette for scientific publications
ALGORITHM_COLORS = {
    "Stucki": "#2c3e50",           # Dark blue-gray
    "Floyd-Steinberg": "#7f8c8d",  # Medium gray
    "Jarvis": "#95a5a6",           # Light gray
    "Standard": "#bdc3c7",         # Lighter gray
}

# Algorithm order for all plots
ALGORITHM_ORDER = ['Stucki', 'Floyd-Steinberg', 'Jarvis', 'Standard']

# Sequential colormap for heatmaps
HEATMAP_CMAP = "YlOrRd"

# Diverging colormap for correlations
CORRELATION_CMAP = "RdBu_r"

# Standard figure sizes (width, height) in inches
FIGSIZE_SINGLE = (5, 4)      # Single panel
FIGSIZE_WIDE = (7, 5)        # Wide single panel
FIGSIZE_DOUBLE = (10, 4)     # Two panels side by side
FIGSIZE_MULTI = (12, 8)      # Multi-panel figure
FIGSIZE_SQUARE = (5, 5)      # Square figure


def get_algorithm_color(alg_name):
    """Get color for algorithm, handling name variations."""
    name = ALGORITHM_NAME_MAP.get(alg_name, alg_name)
    name = name.replace("_", " ").title()
    return ALGORITHM_COLORS.get(name, "#7f7f7f")


def apply_axis_style(ax, grid=True, spines=True):
    """Apply consistent styling to axis."""
    if grid:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    if spines:
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
    
    # Bold tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


def add_significance_annotation(ax, x1, x2, y, p_value, y_offset=0.02):
    """Add significance bracket between two groups."""
    if p_value >= 0.05:
        return
    
    stars = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else '*')
    
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bracket_height = y + y_offset * y_range
    
    ax.plot([x1, x1, x2, x2], 
            [y, bracket_height, bracket_height, y],
            color='black', linewidth=1.0)
    ax.text((x1 + x2) / 2, bracket_height, stars,
            ha='center', va='bottom', fontweight='bold', fontsize=FONT_SIZE)


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
        # Homogeneity metrics
        "coefficient_of_variation": "Coefficient of Variation (%)",
        "regional_cv": "Regional CV (%)",
        "intensity_entropy": "Intensity Entropy",
        "local_variance_std": "Local Variance Std",
        "uniformity_index": "Uniformity Index",
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
# 5.5) Homogeneity Analysis (Low Magnification)
# ============================================================

def compute_homogeneity_metrics(img):
    """
    Compute spatial homogeneity/uniformity metrics for low magnification images.
    
    These metrics assess macro-scale uniformity of the sample, useful for
    evaluating processing consistency and sample quality.
    
    Applicable to low magnifications (250-1,000×) where overall sample
    uniformity is visible.
    
    Args:
        img: Preprocessed grayscale image (uint8)
        
    Returns:
        dict: Homogeneity metrics
    """
    from scipy.stats import entropy
    
    # 1. Coefficient of Variation (CV) - global uniformity
    # Lower CV = more homogeneous
    mean_intensity = img.mean()
    std_intensity = img.std()
    cv = (std_intensity / mean_intensity * 100) if mean_intensity > 0 else 0
    
    # 2. Regional Uniformity - divide into grid and measure variation
    # Divide image into 4×4 = 16 regions
    h, w = img.shape
    region_h = h // 4
    region_w = w // 4
    
    region_means = []
    for i in range(4):
        for j in range(4):
            region = img[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            if region.size > 0:
                region_means.append(region.mean())
    
    region_means = np.array(region_means)
    regional_cv = (np.std(region_means) / np.mean(region_means) * 100) if np.mean(region_means) > 0 else 0
    
    # 3. Intensity Histogram Uniformity
    # Entropy of intensity distribution (lower = more uniform)
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()  # normalize
    intensity_entropy = entropy(hist + 1e-10)  # add small value to avoid log(0)
    
    # 4. Local Variance - standard deviation of local variances
    # High values indicate non-uniform texture across image
    from scipy.ndimage import generic_filter
    
    def local_var(values):
        return np.var(values)
    
    # 11×11 local window variance
    local_variances = generic_filter(img.astype(float), local_var, size=11)
    local_var_std = np.std(local_variances)
    
    # 5. Uniformity Index (0-1, higher = more uniform)
    # Based on histogram evenness
    hist_normalized = hist[hist > 0]  # remove zeros
    uniformity_index = 1.0 / len(hist_normalized) * np.sum(hist_normalized ** 2)
    
    return {
        'coefficient_of_variation': cv,
        'regional_cv': regional_cv,
        'intensity_entropy': intensity_entropy,
        'local_variance_std': local_var_std,
        'uniformity_index': uniformity_index,
    }


# ============================================================
# 6) GLCM
# ============================================================

def compute_glcm_features(img):
    """
    Compute rotation-invariant GLCM texture features.
    
    Calculates GLCM in 4 directions (0°, 45°, 90°, 135°) and averages
    to obtain rotation-invariant texture descriptors.
    
    Distance: 5 pixels (chosen to capture medium-range texture patterns
    while maintaining computational efficiency and statistical reliability)
    
    Args:
        img: Grayscale image (numpy array)
        
    Returns:
        dict: GLCM features (contrast, correlation, energy, homogeneity)
    """
    img = img.astype(np.float32)
    imin, imax = img.min(), img.max()
    if imax > imin:
        img = (img - imin) / (imax - imin)
    else:
        img = np.zeros_like(img)
    img = (img * 255).astype(np.uint8)
    
    # Compute GLCM in 4 directions for rotation invariance
    # Distance=5: captures texture patterns at medium scale
    # Angles: 0°, 45°, 90°, 135° (standard for rotation invariance)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img, [5], angles, 256, symmetric=True, normed=True)
    
    # Average across all directions for rotation-invariant features
    return {
        "contrast": graycoprops(glcm, "contrast")[0, :].mean(),
        "correlation": graycoprops(glcm, "correlation")[0, :].mean(),
        "energy": graycoprops(glcm, "energy")[0, :].mean(),
        "homogeneity": graycoprops(glcm, "homogeneity")[0, :].mean(),
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
                
                # Initialize homogeneity metrics (all NaN by default)
                homogeneity = {
                    'coefficient_of_variation': np.nan,
                    'regional_cv': np.nan,
                    'intensity_entropy': np.nan,
                    'local_variance_std': np.nan,
                    'uniformity_index': np.nan,
                }

                # Low magnification: homogeneity analysis
                if mag <= 1000:
                    homogeneity = compute_homogeneity_metrics(img)

                # Porosity analysis (1000x - 25000x)
                if 1000 <= mag <= 25000:
                    porosity = compute_porosity(img)
                    pores = compute_pore_size(img)
                    if len(pores):
                        pore_area_um2 = pores.mean() * micron_per_pixel ** 2
                        pore_diam_um = np.sqrt(4 * pore_area_um2 / np.pi)

                # Thickness and connectivity (1000x+)
                if mag >= 1000:
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
                    **homogeneity,  # Add homogeneity metrics
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
    """Create publication-quality 3D bar plot for metric comparison."""
    df = pd.read_csv(csv_path)
    df = df[(df.magnification >= mn) & (df.magnification <= mx)]
    if df.empty:
        return

    algs = df.algorithm.unique()
    mags = sorted(df.magnification.unique())

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    # Use professional colormap
    cmap = matplotlib.colormaps["viridis"].resampled(len(mags))

    bar_width = 0.5
    bar_depth = 0.5
    
    for j, m in enumerate(mags):
        for i, a in enumerate(algs):
            v = df[(df.algorithm == a) &
                   (df.magnification == m)][metric].mean()
            if not np.isnan(v):
                ax.bar3d(i - bar_width/2, j - bar_depth/2, 0, 
                        bar_width, bar_depth, v,
                        color=cmap(j), alpha=0.85, 
                        edgecolor='black', linewidth=0.3)

    ax.set_xticks(range(len(algs)))
    xticklabels = [
        ALGORITHM_NAME_MAP.get(a, a).replace("_", " ").title()
        for a in algs
    ]
    ax.set_xticklabels(xticklabels, rotation=30, ha='right', fontweight="bold")
    ax.set_yticks(range(len(mags)))
    ax.set_yticklabels([f"{int(m):,}×" for m in mags], fontweight="bold")
    
    ax.set_xlabel("Algorithm", fontweight="bold", labelpad=10)
    ax.set_ylabel("Magnification", fontweight="bold", labelpad=10)
    ax.set_zlabel(prettify_metric_name(metric), fontweight="bold", labelpad=10)

    # Improve viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Style tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, f"3Dbar_{metric}_{mn}-{mx}.png"),
        dpi=DPI, bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 9) PCA + t-SNE
# ============================================================

def run_dimensionality_reduction(csv_path):
    """Create publication-quality PCA and t-SNE visualizations."""
    df = pd.read_csv(csv_path)
    X = df.select_dtypes(float).fillna(0).values
    X = StandardScaler().fit_transform(X)

    # --- PCA ---
    pca = PCA(2)
    Xp = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    # Color by algorithm if available, otherwise by magnification
    if 'algorithm' in df.columns:
        algorithms = df['algorithm'].unique()
        for alg in algorithms:
            mask = df['algorithm'] == alg
            display_name = ALGORITHM_NAME_MAP.get(alg, alg).replace("_", " ").title()
            color = get_algorithm_color(alg)
            ax.scatter(Xp[mask, 0], Xp[mask, 1], 
                      c=color, label=display_name,
                      s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.legend(loc='best', framealpha=0.9)
    else:
        sc = ax.scatter(Xp[:, 0], Xp[:, 1],
                       c=df.magnification, cmap="viridis",
                       s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Magnification", fontweight="bold")
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")

    # Add variance explained
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontweight="bold")
    ax.set_title("PCA: Dimensionality Reduction", fontweight="bold")
    
    apply_axis_style(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "PCA.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close()

    # --- t-SNE ---
    tsne = TSNE(2, perplexity=min(30, len(X)//4), random_state=42)
    Xt = tsne.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    if 'algorithm' in df.columns:
        for alg in algorithms:
            mask = df['algorithm'] == alg
            display_name = ALGORITHM_NAME_MAP.get(alg, alg).replace("_", " ").title()
            color = get_algorithm_color(alg)
            ax.scatter(Xt[mask, 0], Xt[mask, 1], 
                      c=color, label=display_name,
                      s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.legend(loc='best', framealpha=0.9)
    else:
        sc = ax.scatter(Xt[:, 0], Xt[:, 1],
                       c=df.magnification, cmap="viridis",
                       s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Magnification", fontweight="bold")
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")

    ax.set_xlabel("t-SNE Dimension 1", fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontweight="bold")
    ax.set_title("t-SNE: Nonlinear Dimensionality Reduction", fontweight="bold")
    
    apply_axis_style(ax)
    plt.tight_layout()
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
        # Homogeneity metrics (low magnification)
        "coefficient_of_variation",
        "regional_cv",
        "intensity_entropy",
        "uniformity_index",
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

            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            
            # Get algorithm order and colors
            alg_order = ['Floyd-Steinberg', 'Jarvis', 'Stucki', 'Standard']
            alg_order = [a for a in alg_order if a in sub['algorithm'].unique()]
            colors = [ALGORITHM_COLORS.get(a, '#7f7f7f') for a in alg_order]
            
            # Create boxplot with algorithm colors
            bp = ax.boxplot([sub[sub['algorithm'] == a][metric].values for a in alg_order],
                           patch_artist=True, labels=alg_order,
                           medianprops=dict(color='black', linewidth=1.5),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2),
                           flierprops=dict(marker='o', markersize=4, alpha=0.5))
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.2)
            
            # Add individual data points
            for i, alg in enumerate(alg_order):
                y_data = sub[sub['algorithm'] == alg][metric].values
                x_jitter = np.random.normal(i + 1, 0.04, size=len(y_data))
                ax.scatter(x_jitter, y_data, color='black', alpha=0.4, s=20, zorder=3)
            
            display_metric = prettify_metric_name(metric)
            ax.set_xlabel("Algorithm", fontweight="bold")
            ax.set_ylabel(display_metric, fontweight="bold")

            # Mean ± std annotation
            stats = sub.groupby("algorithm")[metric].agg(["mean", "std"]).reset_index()
            stats["std"] = stats["std"].fillna(0.0)
            
            for i, alg in enumerate(alg_order):
                row = stats[stats["algorithm"] == alg]
                if row.empty:
                    continue
                m = float(row["mean"].values[0])
                s = float(row["std"].values[0])
                y_max = sub[sub['algorithm'] == alg][metric].max()
                ax.text(i + 1, y_max + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                       f"{m:.2f}±{s:.2f}",
                       ha="center", va="bottom", fontweight="bold",
                       fontsize=FONT_SIZE - 2)
            
            # Y-axis limits
            ylim = auto_ylim(sub[metric])
            if ylim is not None:
                ax.set_ylim(bottom=ylim[0], top=ylim[1] * 1.1)
            
            apply_axis_style(ax)
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

                    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
                    
                    alg_order = ['Floyd-Steinberg', 'Jarvis', 'Stucki', 'Standard']
                    alg_order = [a for a in alg_order if a in sub_mag['algorithm'].unique()]
                    colors = [ALGORITHM_COLORS.get(a, '#7f7f7f') for a in alg_order]
                    
                    bp = ax.boxplot([sub_mag[sub_mag['algorithm'] == a][metric].values for a in alg_order],
                                   patch_artist=True, labels=alg_order,
                                   medianprops=dict(color='black', linewidth=1.5),
                                   whiskerprops=dict(linewidth=1.2),
                                   capprops=dict(linewidth=1.2))
                    
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(1.2)
                    
                    # Add data points
                    for i, alg in enumerate(alg_order):
                        y_data = sub_mag[sub_mag['algorithm'] == alg][metric].values
                        x_jitter = np.random.normal(i + 1, 0.04, size=len(y_data))
                        ax.scatter(x_jitter, y_data, color='black', alpha=0.4, s=20, zorder=3)
                    
                    display_metric = prettify_metric_name(metric)
                    ax.set_xlabel("Algorithm", fontweight="bold")
                    ax.set_ylabel(display_metric, fontweight="bold")
                    ax.set_title(f"{int(mag):,}× Magnification", fontweight="bold")

                    ylim = auto_ylim(sub_mag[metric])
                    if ylim is not None:
                        ax.set_ylim(bottom=ylim[0], top=ylim[1] * 1.05)
                    
                    apply_axis_style(ax)
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
                    # Calculate mean and std
                    stats_df = (
                        sub_common.groupby("algorithm")[metric]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    stats_df["algorithm"] = (
                        stats_df["algorithm"].replace(ALGORITHM_NAME_MAP)
                        .astype(str)
                        .str.replace("_", " ")
                        .str.title()
                    )
                    stats_df["std"] = stats_df["std"].fillna(0)

                    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
                    
                    alg_order = ['Floyd-Steinberg', 'Jarvis', 'Stucki', 'Standard']
                    alg_order = [a for a in alg_order if a in stats_df['algorithm'].unique()]
                    colors = [ALGORITHM_COLORS.get(a, '#7f7f7f') for a in alg_order]
                    
                    x = np.arange(len(alg_order))
                    means = [stats_df[stats_df['algorithm'] == a]['mean'].values[0] for a in alg_order]
                    stds = [stats_df[stats_df['algorithm'] == a]['std'].values[0] for a in alg_order]
                    
                    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                                 edgecolor='black', linewidth=1.2, alpha=0.8,
                                 error_kw={'linewidth': 1.5, 'capthick': 1.5})
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(alg_order)
                    
                    display_metric = prettify_metric_name(metric)
                    ax.set_xlabel("Algorithm", fontweight="bold")
                    ax.set_ylabel(display_metric, fontweight="bold")
                    ax.set_title("Mean ± SD Across Magnifications", fontweight="bold")

                    # Add value labels on bars
                    for i, (m, s) in enumerate(zip(means, stds)):
                        ax.text(i, m + s + 0.02 * max(means), f"{m:.2f}",
                               ha="center", va="bottom", fontweight="bold",
                               fontsize=FONT_SIZE - 1)

                    ax.set_ylim(bottom=0, top=max(means) * 1.25)
                    apply_axis_style(ax, grid=True)
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

            plt.figure(figsize=(6, 5))
            display_metric = prettify_metric_name(metric)
            ax = sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",  # Professional sequential colormap
                cbar_kws={"label": display_metric},
                annot_kws={"fontweight": "bold", "fontsize": FONT_SIZE-1},
                linewidths=0.5,
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
# 10.5) HOMOGENEITY VISUALIZATIONS
# ============================================================

def visualize_homogeneity(csv_path):
    """
    Generate comprehensive visualizations for homogeneity metrics.
    Creates radar charts, comparison plots, and heatmaps for low-magnification data.
    """
    df = pd.read_csv(csv_path)
    
    homogeneity_metrics = [
        "coefficient_of_variation",
        "regional_cv", 
        "intensity_entropy",
        "local_variance_std",
        "uniformity_index"
    ]
    
    metric_labels = {
        "coefficient_of_variation": "CV (%)",
        "regional_cv": "Regional CV (%)",
        "intensity_entropy": "Entropy",
        "local_variance_std": "Local Var. Std",
        "uniformity_index": "Uniformity"
    }
    
    # Filter for low magnification data (≤1000×)
    if "magnification" in df.columns:
        low_mag_df = df[df["magnification"] <= 1000].copy()
    else:
        low_mag_df = df.copy()
    
    # Check if we have homogeneity data
    available_metrics = [m for m in homogeneity_metrics if m in low_mag_df.columns]
    if not available_metrics or low_mag_df.empty:
        print("  No homogeneity data available for visualization")
        return
    
    low_mag_df = low_mag_df.dropna(subset=available_metrics, how='all')
    if low_mag_df.empty:
        print("  No valid homogeneity data after dropping NaN")
        return
    
    # Create homogeneity subfolder
    homog_dir = os.path.join(FIG_DIR, "homogeneity")
    os.makedirs(homog_dir, exist_ok=True)
    
    # --- 1) Radar Chart: Algorithm Comparison ---
    if "algorithm" in low_mag_df.columns:
        algorithms = low_mag_df["algorithm"].dropna().unique()
        if len(algorithms) >= 2 and len(available_metrics) >= 3:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            # Normalize metrics to 0-1 scale for radar
            normalized_data = {}
            for metric in available_metrics:
                vals = low_mag_df[metric].dropna()
                if len(vals) > 0:
                    min_val, max_val = vals.min(), vals.max()
                    if max_val > min_val:
                        normalized_data[metric] = (vals - min_val) / (max_val - min_val)
                    else:
                        normalized_data[metric] = vals * 0 + 0.5
            
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the polygon
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
            
            for i, alg in enumerate(algorithms):
                alg_data = low_mag_df[low_mag_df["algorithm"] == alg]
                values = []
                for metric in available_metrics:
                    if metric in normalized_data.index if hasattr(normalized_data, 'index') else metric in normalized_data:
                        metric_vals = alg_data[metric].dropna()
                        if len(metric_vals) > 0:
                            # Normalize this algorithm's mean
                            all_vals = low_mag_df[metric].dropna()
                            min_val, max_val = all_vals.min(), all_vals.max()
                            if max_val > min_val:
                                norm_mean = (metric_vals.mean() - min_val) / (max_val - min_val)
                            else:
                                norm_mean = 0.5
                            values.append(norm_mean)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                
                values += values[:1]  # Close polygon
                display_name = ALGORITHM_NAME_MAP.get(alg, alg.replace("_", " ").title())
                ax.plot(angles, values, 'o-', linewidth=2, label=display_name, color=colors[i])
                ax.fill(angles, values, alpha=0.15, color=colors[i])
            
            # Set labels
            labels = [metric_labels.get(m, m) for m in available_metrics]
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontweight='bold')
            ax.set_ylim(0, 1)
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.title("Homogeneity Metrics by Algorithm\n(Normalized)", fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(homog_dir, "radar_algorithm_comparison.png"),
                       dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Saved: radar_algorithm_comparison.png")
    
    # --- 2) Grouped Bar Chart: All Metrics by Algorithm ---
    if "algorithm" in low_mag_df.columns:
        algorithms = low_mag_df["algorithm"].dropna().unique()
        if len(algorithms) >= 2:
            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            axes = axes.flatten()
            
            for idx, metric in enumerate(available_metrics[:5]):
                ax = axes[idx]
                plot_data = low_mag_df[["algorithm", metric]].dropna()
                plot_data["algorithm"] = (
                    plot_data["algorithm"].replace(ALGORITHM_NAME_MAP)
                    .astype(str).str.replace("_", " ").str.title()
                )
                
                if not plot_data.empty:
                    sns.boxplot(data=plot_data, x="algorithm", y=metric, hue="algorithm",
                               ax=ax, palette="Set2", width=0.6, legend=False)
                    sns.stripplot(data=plot_data, x="algorithm", y=metric,
                                 ax=ax, color="black", alpha=0.4, size=4)
                    
                    ax.set_xlabel("")
                    ax.set_ylabel(metric_labels.get(metric, metric), fontweight='bold')
                    ax.tick_params(axis='x', rotation=30)
                    
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontweight('bold')
            
            # Hide unused subplot
            if len(available_metrics) < 6:
                axes[5].axis('off')
            
            plt.suptitle("Homogeneity Metrics Distribution by Algorithm", 
                        fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(homog_dir, "boxplot_homogeneity_metrics.png"),
                       dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Saved: boxplot_homogeneity_metrics.png")
    
    # --- 3) Correlation Heatmap for Homogeneity Metrics ---
    corr_data = low_mag_df[available_metrics].dropna()
    if len(corr_data) >= 5:
        corr_matrix = corr_data.corr(method='spearman')
        
        fig, ax = plt.subplots(figsize=(8, 7))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        labels = [metric_labels.get(m, m) for m in available_metrics]
        corr_matrix_display = corr_matrix.copy()
        corr_matrix_display.index = labels
        corr_matrix_display.columns = labels
        
        sns.heatmap(corr_matrix_display, mask=mask, annot=True, fmt=".2f",
                   cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, ax=ax,
                   annot_kws={"fontsize": 10, "fontweight": "bold"})
        
        ax.set_title("Homogeneity Metrics Correlation", fontweight='bold')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_rotation(45)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(homog_dir, "correlation_homogeneity.png"),
                   dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: correlation_homogeneity.png")
    
    # --- 4) Power-Speed Heatmap for Homogeneity ---
    if "power" in low_mag_df.columns and "speed" in low_mag_df.columns:
        for metric in available_metrics[:3]:  # Top 3 metrics
            pivot_data = low_mag_df.pivot_table(
                values=metric, 
                index="power", 
                columns="speed", 
                aggfunc="mean"
            )
            
            if pivot_data.size >= 4:
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlOrRd",
                           ax=ax, cbar_kws={"label": metric_labels.get(metric, metric)},
                           annot_kws={"fontweight": "bold"})
                
                ax.set_xlabel("Speed (%)", fontweight='bold')
                ax.set_ylabel("Power (%)", fontweight='bold')
                ax.set_title(f"{metric_labels.get(metric, metric)} vs Processing Parameters",
                           fontweight='bold')
                
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
                
                plt.tight_layout()
                plt.savefig(os.path.join(homog_dir, f"heatmap_power_speed_{metric}.png"),
                           dpi=DPI, bbox_inches="tight")
                plt.close()
    
    # --- 5) CV vs Uniformity Scatter Plot ---
    if "coefficient_of_variation" in available_metrics and "uniformity_index" in available_metrics:
        scatter_data = low_mag_df[["coefficient_of_variation", "uniformity_index", "algorithm"]].dropna()
        
        if len(scatter_data) >= 5 and "algorithm" in scatter_data.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            algorithms = scatter_data["algorithm"].unique()
            colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
            
            for i, alg in enumerate(algorithms):
                alg_data = scatter_data[scatter_data["algorithm"] == alg]
                display_name = ALGORITHM_NAME_MAP.get(alg, alg.replace("_", " ").title())
                ax.scatter(alg_data["coefficient_of_variation"], 
                          alg_data["uniformity_index"],
                          c=[colors[i]], label=display_name, s=80, alpha=0.7, 
                          edgecolors='black', linewidths=0.5)
            
            ax.set_xlabel("Coefficient of Variation (%)", fontweight='bold')
            ax.set_ylabel("Uniformity Index", fontweight='bold')
            ax.set_title("CV vs Uniformity by Algorithm", fontweight='bold')
            ax.legend(loc='best')
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(homog_dir, "scatter_cv_uniformity.png"),
                       dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Saved: scatter_cv_uniformity.png")
    
    # --- 6) Bar Chart with Error Bars: Mean ± Std ---
    if "algorithm" in low_mag_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate means and stds
        stats_data = []
        for alg in low_mag_df["algorithm"].unique():
            alg_data = low_mag_df[low_mag_df["algorithm"] == alg]
            display_name = ALGORITHM_NAME_MAP.get(alg, alg.replace("_", " ").title())
            for metric in available_metrics[:4]:
                vals = alg_data[metric].dropna()
                if len(vals) > 0:
                    stats_data.append({
                        "Algorithm": display_name,
                        "Metric": metric_labels.get(metric, metric),
                        "Mean": vals.mean(),
                        "Std": vals.std()
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            # Create grouped bar chart
            metrics_unique = stats_df["Metric"].unique()
            algorithms_unique = stats_df["Algorithm"].unique()
            x = np.arange(len(metrics_unique))
            width = 0.8 / len(algorithms_unique)
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms_unique)))
            
            for i, alg in enumerate(algorithms_unique):
                alg_stats = stats_df[stats_df["Algorithm"] == alg]
                means = [alg_stats[alg_stats["Metric"] == m]["Mean"].values[0] 
                        if len(alg_stats[alg_stats["Metric"] == m]) > 0 else 0 
                        for m in metrics_unique]
                stds = [alg_stats[alg_stats["Metric"] == m]["Std"].values[0] 
                       if len(alg_stats[alg_stats["Metric"] == m]) > 0 else 0 
                       for m in metrics_unique]
                
                offset = (i - len(algorithms_unique)/2 + 0.5) * width
                bars = ax.bar(x + offset, means, width, label=alg, color=colors[i],
                             yerr=stds, capsize=3, error_kw={'linewidth': 1})
            
            ax.set_xlabel("Metric", fontweight='bold')
            ax.set_ylabel("Value", fontweight='bold')
            ax.set_title("Homogeneity Metrics: Mean ± Std by Algorithm", fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_unique, rotation=30, ha='right')
            ax.legend(loc='upper right')
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(homog_dir, "bar_mean_std_comparison.png"),
                       dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Saved: bar_mean_std_comparison.png")
    
    print(f"  Homogeneity visualizations saved to: {homog_dir}")


# ============================================================
# 11) STATISTICAL ANALYSIS (Q1 PUBLICATION)
# ============================================================

def run_statistical_analysis(csv_path):
    """
    Perform comprehensive statistical analysis for Q1 publication.
    
    Compares algorithms across all metrics with:
    - Normality and homogeneity tests
    - ANOVA or Kruskal-Wallis tests
    - Post-hoc pairwise comparisons
    - Effect size calculations
    - Publication-ready tables
    """
    from statistical_analysis import (
        analyze_metric_by_algorithm,
        generate_descriptive_table,
        generate_anova_table,
        save_statistical_report,
        format_significance_stars
    )
    
    df = pd.read_csv(csv_path)
    
    metrics = [
        "porosity_%",
        "avg_pore_area_um2",
        "avg_pore_equiv_diam_um",
        "avg_thickness_um",
        "connectivity_index",
        "fractal_dimension",
        "contrast",
        "correlation",
        "energy",
        "homogeneity",
        # Homogeneity metrics (low magnification)
        "coefficient_of_variation",
        "regional_cv",
        "intensity_entropy",
        "uniformity_index",
    ]
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS - Algorithm Comparison")
    print("="*80 + "\n")
    
    results_dict = {}
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        print(f"\n📊 Analyzing: {prettify_metric_name(metric)}")
        
        # Full dataset analysis
        results = analyze_metric_by_algorithm(df, metric, algorithm_col='algorithm')
        results_dict[metric] = results
        
        # Print summary
        if 'comparison' in results:
            comp = results['comparison']
            print(f"   Test: {comp.get('test', 'N/A')}")
            print(f"   p-value: {comp.get('p_value', np.nan):.4f}")
            print(f"   Effect size: {comp.get('effect_size', np.nan):.3f}")
            print(f"   Significant: {'YES ✓' if comp.get('significant') else 'NO'}")
            
            if comp.get('significant') and results.get('posthoc') is not None:
                print(f"   Post-hoc comparisons: {len(results['posthoc'])} pairs")
    
    # Save comprehensive report
    report_path = os.path.join(_SCRIPT_DIR, "SEM_statistical_report.txt")
    save_statistical_report(results_dict, report_path)
    
    # Generate LaTeX tables
    tables_dir = os.path.join(_SCRIPT_DIR, "SEM_tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    # Descriptive statistics table
    desc_table = generate_descriptive_table(results_dict, latex=True)
    with open(os.path.join(tables_dir, "descriptive_stats.tex"), 'w') as f:
        f.write(desc_table)
    
    # ANOVA results table
    anova_table = generate_anova_table(results_dict, latex=True)
    with open(os.path.join(tables_dir, "anova_results.tex"), 'w') as f:
        f.write(anova_table)
    
    print(f"\n✔ LaTeX tables saved → {tables_dir}/")
    
    # Create correlation matrix
    plot_correlation_matrix(csv_path)
    
    # Create effect size visualization
    plot_effect_sizes(results_dict)
    
    # Add statistical annotations to existing boxplots
    add_statistical_annotations_to_plots(df, results_dict)
    
    return results_dict


def plot_correlation_matrix(csv_path):
    """Create correlation matrix for all metrics."""
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    
    df = pd.read_csv(csv_path)
    
    metrics = [
        "porosity_%", "avg_pore_area_um2", "avg_pore_equiv_diam_um",
        "avg_thickness_um", "connectivity_index", "fractal_dimension",
        "contrast", "correlation", "energy", "homogeneity",
        # Homogeneity metrics
        "coefficient_of_variation", "regional_cv", "intensity_entropy", "uniformity_index"
    ]
    
    # Select available metrics
    available = [m for m in metrics if m in df.columns]
    data = df[available].dropna()
    
    if data.empty or len(available) < 2:
        print("⚠️ Insufficient data for correlation matrix")
        return
    
    # Compute Spearman correlations
    corr_matrix = data.corr(method='spearman')
    
    # Compute p-values
    n = len(data)
    p_matrix = np.ones((len(available), len(available)))
    
    for i in range(len(available)):
        for j in range(i+1, len(available)):
            _, p = spearmanr(data.iloc[:, i], data.iloc[:, j], nan_policy='omit')
            p_matrix[i, j] = p
            p_matrix[j, i] = p
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap with professional colormap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
        annot_kws={"fontsize": FONT_SIZE-2, "fontweight": "bold"}
    )
    
    # Add significance stars
    for i in range(len(available)):
        for j in range(i+1, len(available)):
            if p_matrix[i, j] < 0.05:
                stars = '***' if p_matrix[i, j] < 0.001 else ('**' if p_matrix[i, j] < 0.01 else '*')
                ax.text(j + 0.5, i + 0.7, stars, ha='center', va='center',
                       fontsize=FONT_SIZE-4, color='black', fontweight='bold')
    
    # Format labels
    labels = [prettify_metric_name(m) for m in available]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontweight='bold')
    ax.set_yticklabels(labels, rotation=0, fontweight='bold')
    
    plt.title('Metric Correlations (Spearman ρ)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_matrix.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✔ Correlation matrix saved → {FIG_DIR}/correlation_matrix.png")


def plot_effect_sizes(results_dict):
    """Visualize effect sizes for all metrics."""
    effect_data = []
    
    for metric, results in results_dict.items():
        if 'comparison' in results and 'effect_size' in results['comparison']:
            effect_data.append({
                'Metric': prettify_metric_name(metric),
                'Effect Size': results['comparison']['effect_size'],
                'Test': results['comparison'].get('test_type', 'unknown')
            })
    
    if not effect_data:
        print("⚠️ No effect size data available")
        return
    
    df = pd.DataFrame(effect_data)
    df = df.sort_values('Effect Size', ascending=False)
    
    plt.figure(figsize=(8, 5))
    # Professional color scheme: dark blue for large, medium blue, light blue
    colors = ['#1f4e79' if x >= 0.14 else ('#2e75b6' if x >= 0.06 else '#9dc3e6') 
              for x in df['Effect Size']]
    
    bars = plt.barh(df['Metric'], df['Effect Size'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add reference lines
    plt.axvline(0.01, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    plt.axvline(0.06, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    plt.axvline(0.14, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add labels for reference lines
    y_top = len(df) - 0.5
    plt.text(0.01, y_top, 'Small', fontsize=9, ha='center', va='bottom', color='gray')
    plt.text(0.06, y_top, 'Medium', fontsize=9, ha='center', va='bottom', color='gray')
    plt.text(0.14, y_top, 'Large', fontsize=9, ha='center', va='bottom', color='gray')
    
    # Add values on bars
    for i, (idx, row) in enumerate(df.iterrows()):
        plt.text(row['Effect Size'] + 0.005, i, f"{row['Effect Size']:.3f}",
                va='center', fontweight='bold', fontsize=FONT_SIZE-1)
    
    plt.xlabel('Effect Size (η² or ε²)', fontweight='bold')
    plt.ylabel('Metric', fontweight='bold')
    plt.title('Effect Sizes: Algorithm Comparison', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "effect_sizes.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✔ Effect sizes plot saved → {FIG_DIR}/effect_sizes.png")


def add_statistical_annotations_to_plots(df, results_dict):
    """
    Recreate boxplots with statistical significance annotations.
    
    Adds brackets and stars (*, **, ***) for significant pairwise comparisons.
    """
    from statistical_analysis import format_significance_stars
    
    metrics = [m for m in results_dict.keys() if 'posthoc' in results_dict[m] and 
               results_dict[m]['posthoc'] is not None]
    
    for metric in metrics:
        results = results_dict[metric]
        posthoc = results['posthoc']
        
        if posthoc is None or posthoc.empty:
            continue
        
        # Filter significant comparisons
        sig_pairs = posthoc[posthoc['significant'] == True]
        
        if sig_pairs.empty:
            continue
        
        # Prepare data
        sub = df[["algorithm", metric]].dropna().copy()
        sub["algorithm"] = (
            sub["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        if sub.empty:
            continue
        
        # Create publication-quality plot
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        
        alg_order = ['Floyd-Steinberg', 'Jarvis', 'Stucki', 'Standard']
        alg_order = [a for a in alg_order if a in sub['algorithm'].unique()]
        colors = [ALGORITHM_COLORS.get(a, '#7f7f7f') for a in alg_order]
        
        bp = ax.boxplot([sub[sub['algorithm'] == a][metric].values for a in alg_order],
                       patch_artist=True, labels=alg_order,
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(linewidth=1.2),
                       capprops=dict(linewidth=1.2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        
        # Add data points
        for i, alg in enumerate(alg_order):
            y_data = sub[sub['algorithm'] == alg][metric].values
            x_jitter = np.random.normal(i + 1, 0.04, size=len(y_data))
            ax.scatter(x_jitter, y_data, color='black', alpha=0.4, s=20, zorder=3)
        
        display_metric = prettify_metric_name(metric)
        ax.set_xlabel("Algorithm", fontweight="bold")
        ax.set_ylabel(display_metric, fontweight="bold")
        ax.set_title(f"{display_metric} - Statistical Comparison", fontweight="bold")
        
        # Get y-axis limits
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Add significance brackets
        alg_positions = {alg: i + 1 for i, alg in enumerate(alg_order)}
        
        bracket_height = y_max + 0.05 * y_range
        bracket_offset = 0.1 * y_range
        
        for idx, (_, row) in enumerate(sig_pairs.iterrows()):
            g1 = row['group1']
            g1 = ALGORITHM_NAME_MAP.get(g1, g1).replace("_", " ").title()
            g2 = row['group2']
            g2 = ALGORITHM_NAME_MAP.get(g2, g2).replace("_", " ").title()
            
            if g1 not in alg_positions or g2 not in alg_positions:
                continue
            
            x1 = alg_positions[g1]
            x2 = alg_positions[g2]
            
            y = bracket_height + idx * bracket_offset
            
            # Draw bracket
            ax.plot([x1, x1, x2, x2], [y, y+0.02*y_range, y+0.02*y_range, y],
                   'k-', linewidth=1.2)
            
            # Add stars
            stars = format_significance_stars(row.get('p_corrected', row['p_value']))
            ax.text((x1 + x2) / 2, y + 0.025*y_range, stars,
                   ha='center', va='bottom', fontweight='bold', fontsize=FONT_SIZE)
        
        # Adjust y-limit
        if not sig_pairs.empty:
            ax.set_ylim(y_min, bracket_height + len(sig_pairs) * bracket_offset + 0.1 * y_range)
        
        apply_axis_style(ax)
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR, f"box_algorithm_{metric}_with_stats.png"),
            dpi=DPI,
            bbox_inches="tight"
        )
        plt.close()
        
        print(f"✔ Statistical boxplot saved → box_algorithm_{metric}_with_stats.png")


# ============================================================
# 11.5) FINAL PUBLICATION FIGURES
# ============================================================

def generate_final_publication_figures(csv_path):
    """
    Generate final publication-ready summary figures for all metrics.
    
    For each metric, averages ALL available values across all magnifications
    where that metric was calculated (i.e., non-NaN values).
    
    Saves bar charts with error bars (mean ± SD) to SEM_figures/final/
    """
    df = pd.read_csv(csv_path)
    
    # Create final directory
    final_dir = os.path.join(FIG_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING FINAL PUBLICATION FIGURES")
    print("="*80)
    
    # Define all metrics with their display properties
    metric_config = {
        # Porosity metrics
        "porosity_%": {"unit": "%", "label": "Porosity"},
        "avg_pore_area_um2": {"unit": "µm²", "label": "Average Pore Area"},
        "avg_pore_equiv_diam_um": {"unit": "µm", "label": "Average Pore Diameter"},
        
        # Fine structure metrics
        "avg_thickness_um": {"unit": "µm", "label": "Average Fiber Thickness"},
        "connectivity_index": {"unit": "", "label": "Connectivity Index"},
        "fractal_dimension": {"unit": "", "label": "Fractal Dimension"},
        
        # Homogeneity metrics
        "coefficient_of_variation": {"unit": "%", "label": "Coefficient of Variation"},
        "regional_cv": {"unit": "%", "label": "Regional CV"},
        "intensity_entropy": {"unit": "", "label": "Intensity Entropy"},
        "local_variance_std": {"unit": "", "label": "Local Variance Std"},
        "uniformity_index": {"unit": "", "label": "Uniformity Index"},
        
        # GLCM texture metrics
        "contrast": {"unit": "", "label": "GLCM Contrast"},
        "correlation": {"unit": "", "label": "GLCM Correlation"},
        "energy": {"unit": "", "label": "GLCM Energy"},
        "homogeneity": {"unit": "", "label": "GLCM Homogeneity"},
    }
    
    # Algorithm order: Stucki, Floyd, Jarvis, Standard
    alg_order = ALGORITHM_ORDER
    
    generated_count = 0
    
    for metric, config in metric_config.items():
        if metric not in df.columns:
            continue
        
        # Get ALL non-NaN data for this metric (no magnification filtering)
        metric_data = df[["algorithm", metric]].dropna().copy()
        
        if metric_data.empty or len(metric_data) < 4:
            continue
        
        # Find which magnifications have this metric
        if "magnification" in df.columns:
            mags_with_data = df[df[metric].notna()]["magnification"].unique()
            mags_with_data = sorted(mags_with_data)
            if len(mags_with_data) > 0:
                mag_str = f"{int(min(mags_with_data)):,}–{int(max(mags_with_data)):,}×"
            else:
                mag_str = "All"
        else:
            mag_str = "All"
        
        # Standardize algorithm names
        metric_data["algorithm"] = (
            metric_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        # Calculate statistics per algorithm (averaging across ALL magnifications)
        stats = metric_data.groupby("algorithm")[metric].agg(["mean", "std", "count"]).reset_index()
        stats["std"] = stats["std"].fillna(0)
        
        # Filter to available algorithms and maintain order
        available_algs = [a for a in alg_order if a in stats["algorithm"].values]
        if len(available_algs) < 2:
            continue
        
        # Reorder stats
        stats = stats.set_index("algorithm").loc[available_algs].reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))
        
        x = np.arange(len(available_algs))
        bar_colors = [ALGORITHM_COLORS.get(a, '#4a4a4a') for a in available_algs]
        
        # Bar chart with error bars
        bars = ax.bar(x, stats["mean"], yerr=stats["std"], 
                     capsize=6, color=bar_colors, edgecolor='black', 
                     linewidth=1.5, alpha=0.9,
                     error_kw={'linewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})
        
        # Add value labels on top of bars (mean ± std)
        for i, (mean, std) in enumerate(zip(stats["mean"], stats["std"])):
            y_pos = mean + std + 0.05 * (stats["mean"] + stats["std"]).max()
            ax.text(i, y_pos, f"{mean:.2f}±{std:.2f}", ha='center', va='bottom', 
                   fontweight='bold', fontsize=FONT_SIZE - 1)
        
        # Axis labels and title
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold')
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_xlabel("Algorithm", fontweight='bold')
        
        # Title with magnification range
        ax.set_title(f"{config['label']} (Avg: {mag_str})", fontweight='bold')
        
        # Y-axis: start from 0, add padding for labels
        y_max = (stats["mean"] + stats["std"]).max()
        ax.set_ylim(0, y_max * 1.35)
        
        # Grid and styling
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        # Add sample size annotation
        n_total = stats["count"].sum()
        ax.text(0.98, 0.02, f"n = {int(n_total)}", transform=ax.transAxes,
               ha='right', va='bottom', fontsize=FONT_SIZE-2, style='italic')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"final_{metric}.png"
        filepath = os.path.join(final_dir, filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        generated_count += 1
        print(f"  ✔ {config['label']:25s} → {filename} (n={int(n_total)}, mag: {mag_str})")
    
    # Generate combined multi-panel figure
    _generate_combined_final_figure(df, final_dir, metric_config, alg_order)
    
    print(f"\n✔ Generated {generated_count} final figures → {final_dir}")
    return final_dir


def _generate_combined_final_figure(df, final_dir, metric_config, alg_order):
    """Generate a combined multi-panel figure with key metrics."""
    
    # Select key metrics for combined figure
    key_metrics = [
        "porosity_%",
        "avg_thickness_um", 
        "connectivity_index",
        "coefficient_of_variation"
    ]
    
    # Filter to available metrics
    available_metrics = [m for m in key_metrics if m in df.columns and df[m].notna().sum() > 0]
    if len(available_metrics) < 2:
        return
    
    n_metrics = len(available_metrics)
    ncols = 2
    nrows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4*nrows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    
    # Use ALGORITHM_ORDER for combined figure
    alg_order = ALGORITHM_ORDER
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        config = metric_config[metric]
        
        # Get ALL non-NaN data (no magnification filtering)
        metric_data = df[["algorithm", metric]].dropna().copy()
        
        if metric_data.empty:
            ax.set_visible(False)
            continue
        
        metric_data["algorithm"] = (
            metric_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        stats = metric_data.groupby("algorithm")[metric].agg(["mean", "std"]).reset_index()
        stats["std"] = stats["std"].fillna(0)
        
        available_algs = [a for a in alg_order if a in stats["algorithm"].values]
        if not available_algs:
            ax.set_visible(False)
            continue
        
        stats = stats.set_index("algorithm").loc[available_algs].reset_index()
        
        x = np.arange(len(available_algs))
        bar_colors = [ALGORITHM_COLORS.get(a, '#4a4a4a') for a in available_algs]
        
        ax.bar(x, stats["mean"], yerr=stats["std"], capsize=4, 
               color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.9,
               error_kw={'linewidth': 1.2, 'capthick': 1.2})
        
        # Add value labels (mean ± std) with adaptive precision
        for i, (mean, std) in enumerate(zip(stats["mean"], stats["std"])):
            y_pos = mean + std + 0.05 * (stats["mean"] + stats["std"]).max()
            # Use more decimals for small values
            if mean < 1:
                label = f"{mean:.2f}±{std:.2f}"
            else:
                label = f"{mean:.1f}±{std:.1f}"
            ax.text(i, y_pos, label, ha='center', va='bottom', 
                   fontweight='bold', fontsize=FONT_SIZE - 3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold', fontsize=FONT_SIZE-1)
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        
        # Panel label (Times New Roman, 12pt)
        ax.text(-0.12, 1.05, panel_labels[idx], transform=ax.transAxes,
               fontweight='bold', fontsize=FONT_SIZE, fontfamily=FONT_FAMILY)
        
        ax.set_ylim(0, (stats["mean"] + stats["std"]).max() * 1.3)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    # Hide unused axes
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(final_dir, "combined_key_metrics.png"), 
               dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(final_dir, "combined_key_metrics.pdf"), 
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✔ Combined figure       → combined_key_metrics.png/pdf")


# ============================================================
# 11.5) 1000x ONLY ANALYSIS
# ============================================================

def analyze_1000x_only(csv_path):
    """
    Generate metrics and figures for 1000x magnification ONLY.
    
    Creates a separate folder '1000x' in SEM_figures with:
    - Bar charts for each metric
    - Combined multi-panel figure
    - Statistical summary
    
    This allows direct comparison at a single magnification level.
    """
    df = pd.read_csv(csv_path)
    
    # Filter for 1000x only
    if "magnification" not in df.columns:
        print("⚠️ No magnification column found - skipping 1000x analysis")
        return
    
    df_1000x = df[df["magnification"] == 1000].copy()
    
    if df_1000x.empty:
        print("⚠️ No 1000x data found - skipping 1000x analysis")
        return
    
    # Create 1000x output directory
    output_dir = os.path.join(FIG_DIR, "1000x")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("1000x MAGNIFICATION ANALYSIS")
    print("="*80)
    print(f"  Total samples at 1000x: n = {len(df_1000x)}")
    
    # Define all metrics to analyze
    metric_config = {
        # Porosity metrics
        "porosity_%": {"unit": "%", "label": "Porosity"},
        "avg_pore_area_um2": {"unit": "µm²", "label": "Average Pore Area"},
        "avg_pore_equiv_diam_um": {"unit": "µm", "label": "Average Pore Diameter"},
        
        # Fine structure metrics
        "avg_thickness_um": {"unit": "µm", "label": "Average Fiber Thickness"},
        "connectivity_index": {"unit": "", "label": "Connectivity Index"},
        "fractal_dimension": {"unit": "", "label": "Fractal Dimension"},
        
        # Homogeneity metrics
        "coefficient_of_variation": {"unit": "%", "label": "Coefficient of Variation"},
        "regional_cv": {"unit": "%", "label": "Regional CV"},
        "intensity_entropy": {"unit": "", "label": "Intensity Entropy"},
        "local_variance_std": {"unit": "", "label": "Local Variance Std"},
        "uniformity_index": {"unit": "", "label": "Uniformity Index"},
        
        # GLCM texture metrics
        "contrast": {"unit": "", "label": "GLCM Contrast"},
        "correlation": {"unit": "", "label": "GLCM Correlation"},
        "energy": {"unit": "", "label": "GLCM Energy"},
        "homogeneity": {"unit": "", "label": "GLCM Homogeneity"},
    }
    
    alg_order = ALGORITHM_ORDER
    generated_count = 0
    
    # Show sample counts per algorithm
    print("\n  Samples per algorithm at 1000x:")
    for alg in df_1000x["algorithm"].unique():
        n = len(df_1000x[df_1000x["algorithm"] == alg])
        print(f"    {alg}: n = {n}")
    
    print("\n  Generating figures...")
    
    # Generate individual metric figures
    for metric, config in metric_config.items():
        if metric not in df_1000x.columns:
            continue
        
        metric_data = df_1000x[["algorithm", metric]].dropna().copy()
        
        if metric_data.empty or len(metric_data) < 2:
            continue
        
        # Standardize algorithm names
        metric_data["algorithm"] = (
            metric_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        # Calculate statistics per algorithm
        stats = metric_data.groupby("algorithm")[metric].agg(["mean", "std", "count"]).reset_index()
        stats["std"] = stats["std"].fillna(0)
        
        # Filter to available algorithms and maintain order
        available_algs = [a for a in alg_order if a in stats["algorithm"].values]
        if len(available_algs) < 2:
            continue
        
        stats = stats.set_index("algorithm").loc[available_algs].reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))
        
        x = np.arange(len(available_algs))
        bar_colors = [ALGORITHM_COLORS.get(a, '#4a4a4a') for a in available_algs]
        
        # Bar chart with error bars
        bars = ax.bar(x, stats["mean"], yerr=stats["std"],
                     capsize=6, color=bar_colors, edgecolor='black',
                     linewidth=1.5, alpha=0.9,
                     error_kw={'linewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})
        
        # Add value labels on top of bars
        for i, (mean, std) in enumerate(zip(stats["mean"], stats["std"])):
            y_pos = mean + std + 0.05 * (stats["mean"] + stats["std"]).max()
            # Use scientific notation for very small values
            if abs(mean) < 0.001 and mean != 0:
                label = f"{mean:.1e}"
            else:
                label = f"{mean:.2f}±{std:.2f}"
            ax.text(i, y_pos, label, ha='center', va='bottom',
                   fontweight='bold', fontsize=FONT_SIZE - 1)
        
        # Axis labels and title
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold')
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_xlabel("Algorithm", fontweight='bold')
        ax.set_title(f"{config['label']} (1000× Only)", fontweight='bold')
        
        # Y-axis: start from 0, add padding
        y_max = (stats["mean"] + stats["std"]).max()
        ax.set_ylim(0, y_max * 1.35)
        
        # Grid and styling
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        # Add sample size annotation
        n_total = stats["count"].sum()
        ax.text(0.98, 0.02, f"n = {int(n_total)}", transform=ax.transAxes,
               ha='right', va='bottom', fontsize=FONT_SIZE-2, style='italic')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"1000x_{metric}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        generated_count += 1
        print(f"    ✔ {config['label']:25s} → {filename}")
    
    # Generate combined multi-panel figure for 1000x
    _generate_combined_1000x_figure(df_1000x, output_dir, metric_config, alg_order)
    
    # Generate statistical summary for 1000x
    _generate_1000x_stats_summary(df_1000x, output_dir, metric_config)
    
    print(f"\n✔ Generated {generated_count} figures for 1000x → {output_dir}")
    return output_dir


def _generate_combined_1000x_figure(df, output_dir, metric_config, alg_order):
    """Generate combined multi-panel figure for 1000x data."""
    
    # Select key metrics for combined figure (ones likely available at 1000x)
    key_metrics = [
        "porosity_%",
        "avg_pore_area_um2",
        "coefficient_of_variation",
        "regional_cv",
        "intensity_entropy",
        "uniformity_index"
    ]
    
    # Filter to available metrics
    available_metrics = [m for m in key_metrics if m in df.columns and df[m].notna().sum() > 0]
    if len(available_metrics) < 2:
        return
    
    n_metrics = len(available_metrics)
    ncols = 2
    nrows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4*nrows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        config = metric_config[metric]
        
        metric_data = df[["algorithm", metric]].dropna().copy()
        
        if metric_data.empty:
            ax.set_visible(False)
            continue
        
        metric_data["algorithm"] = (
            metric_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        stats = metric_data.groupby("algorithm")[metric].agg(["mean", "std"]).reset_index()
        stats["std"] = stats["std"].fillna(0)
        
        available_algs = [a for a in alg_order if a in stats["algorithm"].values]
        if not available_algs:
            ax.set_visible(False)
            continue
        
        stats = stats.set_index("algorithm").loc[available_algs].reset_index()
        
        x = np.arange(len(available_algs))
        bar_colors = [ALGORITHM_COLORS.get(a, '#4a4a4a') for a in available_algs]
        
        ax.bar(x, stats["mean"], yerr=stats["std"], capsize=4,
               color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.9,
               error_kw={'linewidth': 1.2, 'capthick': 1.2})
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(stats["mean"], stats["std"])):
            y_pos = mean + std + 0.05 * (stats["mean"] + stats["std"]).max()
            # Use scientific notation for very small values
            if abs(mean) < 0.001 and mean != 0:
                label = f"{mean:.1e}"
            elif mean < 1:
                label = f"{mean:.2f}±{std:.2f}"
            else:
                label = f"{mean:.1f}±{std:.1f}"
            ax.text(i, y_pos, label, ha='center', va='bottom',
                   fontweight='bold', fontsize=FONT_SIZE - 3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold', fontsize=FONT_SIZE-1)
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        
        # Panel label
        ax.text(-0.12, 1.05, panel_labels[idx], transform=ax.transAxes,
               fontweight='bold', fontsize=FONT_SIZE, fontfamily=FONT_FAMILY)
        
        ax.set_ylim(0, (stats["mean"] + stats["std"]).max() * 1.3)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    # Hide unused axes
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("1000× Magnification Analysis", fontweight='bold', fontsize=FONT_SIZE+2, y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "1000x_combined.png"),
               dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, "1000x_combined.pdf"),
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✔ Combined figure       → 1000x_combined.png/pdf")


def _generate_1000x_stats_summary(df, output_dir, metric_config):
    """Generate statistical summary table for 1000x data."""
    
    summary_rows = []
    
    for metric, config in metric_config.items():
        if metric not in df.columns:
            continue
        
        for alg in df["algorithm"].unique():
            alg_data = df[df["algorithm"] == alg][metric].dropna()
            if len(alg_data) > 0:
                summary_rows.append({
                    "Metric": config["label"],
                    "Unit": config["unit"],
                    "Algorithm": ALGORITHM_NAME_MAP.get(alg, alg),
                    "n": len(alg_data),
                    "Mean": alg_data.mean(),
                    "Std": alg_data.std(),
                    "Min": alg_data.min(),
                    "Max": alg_data.max(),
                    "Median": alg_data.median()
                })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, "1000x_statistics.csv")
        summary_df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"    ✔ Statistics summary   → 1000x_statistics.csv")
        
        # Also generate a readable text report
        txt_path = os.path.join(output_dir, "1000x_report.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("1000× MAGNIFICATION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total samples: n = {len(df)}\n\n")
            
            f.write("Samples per algorithm:\n")
            for alg in df["algorithm"].unique():
                n = len(df[df["algorithm"] == alg])
                f.write(f"  - {ALGORITHM_NAME_MAP.get(alg, alg)}: n = {n}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("METRICS SUMMARY (Mean ± SD)\n")
            f.write("-"*80 + "\n\n")
            
            for metric, config in metric_config.items():
                if metric not in df.columns or df[metric].isna().all():
                    continue
                
                f.write(f"\n{config['label']}")
                if config['unit']:
                    f.write(f" ({config['unit']})")
                f.write(":\n")
                
                for alg in df["algorithm"].unique():
                    alg_data = df[df["algorithm"] == alg][metric].dropna()
                    if len(alg_data) > 0:
                        alg_name = ALGORITHM_NAME_MAP.get(alg, alg)
                        mean_val = alg_data.mean()
                        std_val = alg_data.std()
                        # Use scientific notation for very small values
                        if abs(mean_val) < 0.001 and mean_val != 0:
                            f.write(f"  {alg_name:20s}: {mean_val:10.2e} ± {std_val:9.2e} (n={len(alg_data)})\n")
                        else:
                            f.write(f"  {alg_name:20s}: {mean_val:8.3f} ± {std_val:7.3f} (n={len(alg_data)})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"    ✔ Text report          → 1000x_report.txt")


# ============================================================
# 11.6) 25000x ONLY ANALYSIS
# ============================================================

def analyze_25000x_only(csv_path):
    """
    Generate metrics and figures for 25000x magnification ONLY.
    
    Creates a separate folder '25000x' in SEM_figures with:
    - Bar charts for each metric
    - Combined multi-panel figure
    - Statistical summary
    
    This allows direct comparison at a single magnification level.
    """
    df = pd.read_csv(csv_path)
    
    # Filter for 25000x only
    if "magnification" not in df.columns:
        print("⚠️ No magnification column found - skipping 25000x analysis")
        return
    
    df_25000x = df[df["magnification"] == 25000].copy()
    
    if df_25000x.empty:
        print("⚠️ No 25000x data found - skipping 25000x analysis")
        return
    
    # Create 25000x output directory
    output_dir = os.path.join(FIG_DIR, "25000x")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("25000x MAGNIFICATION ANALYSIS")
    print("="*80)
    print(f"  Total samples at 25000x: n = {len(df_25000x)}")
    
    # Define all metrics to analyze
    metric_config = {
        # Porosity metrics
        "porosity_%": {"unit": "%", "label": "Porosity"},
        "avg_pore_area_um2": {"unit": "µm²", "label": "Average Pore Area"},
        "avg_pore_equiv_diam_um": {"unit": "µm", "label": "Average Pore Diameter"},
        
        # Fine structure metrics
        "avg_thickness_um": {"unit": "µm", "label": "Average Fiber Thickness"},
        "connectivity_index": {"unit": "", "label": "Connectivity Index"},
        "fractal_dimension": {"unit": "", "label": "Fractal Dimension"},
        
        # GLCM texture metrics
        "contrast": {"unit": "", "label": "GLCM Contrast"},
        "correlation": {"unit": "", "label": "GLCM Correlation"},
        "energy": {"unit": "", "label": "GLCM Energy"},
        "homogeneity": {"unit": "", "label": "GLCM Homogeneity"},
    }
    
    alg_order = ALGORITHM_ORDER
    generated_count = 0
    
    # Show sample counts per algorithm
    print("\n  Samples per algorithm at 25000x:")
    for alg in df_25000x["algorithm"].unique():
        n = len(df_25000x[df_25000x["algorithm"] == alg])
        print(f"    {alg}: n = {n}")
    
    print("\n  Generating figures...")
    
    # Generate individual metric figures
    for metric, config in metric_config.items():
        if metric not in df_25000x.columns:
            continue
        
        metric_data = df_25000x[["algorithm", metric]].dropna().copy()
        
        if metric_data.empty or len(metric_data) < 2:
            continue
        
        # Standardize algorithm names
        metric_data["algorithm"] = (
            metric_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        # Calculate statistics per algorithm
        stats = metric_data.groupby("algorithm")[metric].agg(["mean", "std", "count"]).reset_index()
        stats["std"] = stats["std"].fillna(0)
        
        # Filter to available algorithms and maintain order
        available_algs = [a for a in alg_order if a in stats["algorithm"].values]
        if len(available_algs) < 2:
            continue
        
        stats = stats.set_index("algorithm").loc[available_algs].reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))
        
        x = np.arange(len(available_algs))
        bar_colors = [ALGORITHM_COLORS.get(a, '#4a4a4a') for a in available_algs]
        
        # Bar chart with error bars
        bars = ax.bar(x, stats["mean"], yerr=stats["std"],
                     capsize=6, color=bar_colors, edgecolor='black',
                     linewidth=1.5, alpha=0.9,
                     error_kw={'linewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})
        
        # Add value labels on top of bars
        for i, (mean, std) in enumerate(zip(stats["mean"], stats["std"])):
            y_pos = mean + std + 0.05 * (stats["mean"] + stats["std"]).max()
            # Use scientific notation for very small values
            if abs(mean) < 0.001 and mean != 0:
                label = f"{mean:.1e}"
            else:
                label = f"{mean:.2f}±{std:.2f}"
            ax.text(i, y_pos, label, ha='center', va='bottom',
                   fontweight='bold', fontsize=FONT_SIZE - 1)
        
        # Axis labels and title
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold')
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_xlabel("Algorithm", fontweight='bold')
        ax.set_title(f"{config['label']} (25,000× Only)", fontweight='bold')
        
        # Y-axis: start from 0, add padding
        y_max = (stats["mean"] + stats["std"]).max()
        ax.set_ylim(0, y_max * 1.35)
        
        # Grid and styling
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        # Add sample size annotation
        n_total = stats["count"].sum()
        ax.text(0.98, 0.02, f"n = {int(n_total)}", transform=ax.transAxes,
               ha='right', va='bottom', fontsize=FONT_SIZE-2, style='italic')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"25000x_{metric}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        
        generated_count += 1
        print(f"    ✔ {config['label']:25s} → {filename}")
    
    # Generate combined multi-panel figure for 25000x
    _generate_combined_25000x_figure(df_25000x, output_dir, metric_config, alg_order)
    
    # Generate statistical summary for 25000x
    _generate_25000x_stats_summary(df_25000x, output_dir, metric_config)
    
    print(f"\n✔ Generated {generated_count} figures for 25000x → {output_dir}")
    return output_dir


def _generate_combined_25000x_figure(df, output_dir, metric_config, alg_order):
    """Generate combined multi-panel figure for 25000x data."""
    
    # Select key metrics for combined figure (ones likely available at 25000x)
    key_metrics = [
        "porosity_%",
        "avg_pore_area_um2",
        "avg_thickness_um",
        "connectivity_index",
        "fractal_dimension",
        "contrast"
    ]
    
    # Filter to available metrics
    available_metrics = [m for m in key_metrics if m in df.columns and df[m].notna().sum() > 0]
    if len(available_metrics) < 2:
        return
    
    n_metrics = len(available_metrics)
    ncols = 2
    nrows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4*nrows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        config = metric_config[metric]
        
        metric_data = df[["algorithm", metric]].dropna().copy()
        
        if metric_data.empty:
            ax.set_visible(False)
            continue
        
        metric_data["algorithm"] = (
            metric_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        stats = metric_data.groupby("algorithm")[metric].agg(["mean", "std"]).reset_index()
        stats["std"] = stats["std"].fillna(0)
        
        available_algs = [a for a in alg_order if a in stats["algorithm"].values]
        if not available_algs:
            ax.set_visible(False)
            continue
        
        stats = stats.set_index("algorithm").loc[available_algs].reset_index()
        
        x = np.arange(len(available_algs))
        bar_colors = [ALGORITHM_COLORS.get(a, '#4a4a4a') for a in available_algs]
        
        ax.bar(x, stats["mean"], yerr=stats["std"], capsize=4,
               color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.9,
               error_kw={'linewidth': 1.2, 'capthick': 1.2})
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(stats["mean"], stats["std"])):
            y_pos = mean + std + 0.05 * (stats["mean"] + stats["std"]).max()
            # Use scientific notation for very small values
            if abs(mean) < 0.001 and mean != 0:
                label = f"{mean:.1e}"
            elif mean < 1:
                label = f"{mean:.2f}±{std:.2f}"
            else:
                label = f"{mean:.1f}±{std:.1f}"
            ax.text(i, y_pos, label, ha='center', va='bottom',
                   fontweight='bold', fontsize=FONT_SIZE - 3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold', fontsize=FONT_SIZE-1)
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        
        # Panel label
        ax.text(-0.12, 1.05, panel_labels[idx], transform=ax.transAxes,
               fontweight='bold', fontsize=FONT_SIZE, fontfamily=FONT_FAMILY)
        
        ax.set_ylim(0, (stats["mean"] + stats["std"]).max() * 1.3)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    # Hide unused axes
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("25,000× Magnification Analysis", fontweight='bold', fontsize=FONT_SIZE+2, y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "25000x_combined.png"),
               dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, "25000x_combined.pdf"),
               bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✔ Combined figure       → 25000x_combined.png/pdf")


def _generate_25000x_stats_summary(df, output_dir, metric_config):
    """Generate statistical summary table for 25000x data."""
    
    summary_rows = []
    
    for metric, config in metric_config.items():
        if metric not in df.columns:
            continue
        
        for alg in df["algorithm"].unique():
            alg_data = df[df["algorithm"] == alg][metric].dropna()
            if len(alg_data) > 0:
                summary_rows.append({
                    "Metric": config["label"],
                    "Unit": config["unit"],
                    "Algorithm": ALGORITHM_NAME_MAP.get(alg, alg),
                    "n": len(alg_data),
                    "Mean": alg_data.mean(),
                    "Std": alg_data.std(),
                    "Min": alg_data.min(),
                    "Max": alg_data.max(),
                    "Median": alg_data.median()
                })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, "25000x_statistics.csv")
        summary_df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"    ✔ Statistics summary   → 25000x_statistics.csv")
        
        # Also generate a readable text report
        txt_path = os.path.join(output_dir, "25000x_report.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("25,000× MAGNIFICATION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total samples: n = {len(df)}\n\n")
            
            f.write("Samples per algorithm:\n")
            for alg in df["algorithm"].unique():
                n = len(df[df["algorithm"] == alg])
                f.write(f"  - {ALGORITHM_NAME_MAP.get(alg, alg)}: n = {n}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("METRICS SUMMARY (Mean ± SD)\n")
            f.write("-"*80 + "\n\n")
            
            for metric, config in metric_config.items():
                if metric not in df.columns or df[metric].isna().all():
                    continue
                
                f.write(f"\n{config['label']}")
                if config['unit']:
                    f.write(f" ({config['unit']})")
                f.write(":\n")
                
                for alg in df["algorithm"].unique():
                    alg_data = df[df["algorithm"] == alg][metric].dropna()
                    if len(alg_data) > 0:
                        alg_name = ALGORITHM_NAME_MAP.get(alg, alg)
                        mean_val = alg_data.mean()
                        std_val = alg_data.std()
                        # Use scientific notation for very small values
                        if abs(mean_val) < 0.001 and mean_val != 0:
                            f.write(f"  {alg_name:20s}: {mean_val:10.2e} ± {std_val:9.2e} (n={len(alg_data)})\n")
                        else:
                            f.write(f"  {alg_name:20s}: {mean_val:8.3f} ± {std_val:7.3f} (n={len(alg_data)})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"    ✔ Text report          → 25000x_report.txt")


# ============================================================
# 12) SAMPLE SIZE & REPRODUCIBILITY REPORT
# ============================================================

def generate_sample_size_report(csv_path):
    """
    Generate sample size report for each algorithm and magnification.
    
    Reports n for each group and warns if n < 3.
    """
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*80)
    print("SAMPLE SIZE REPORT")
    print("="*80 + "\n")
    
    # Overall sample sizes by algorithm
    print("Overall Sample Sizes by Algorithm:")
    print("-" * 80)
    algo_counts = df.groupby('algorithm').size()
    for algo, count in algo_counts.items():
        status = "⚠️ LOW" if count < 3 else "✓"
        print(f"  {algo}: n = {count} {status}")
    
    # By magnification
    print("\nSample Sizes by Algorithm × Magnification:")
    print("-" * 80)
    
    if 'magnification' in df.columns:
        mag_counts = df.groupby(['algorithm', 'magnification']).size().unstack(fill_value=0)
        print(mag_counts.to_string())
        
        # Check for low samples
        low_samples = mag_counts[mag_counts < 3]
        if low_samples.any().any():
            print("\n⚠️ Warning: The following groups have n < 3:")
            for algo in low_samples.index:
                for mag in low_samples.columns:
                    if low_samples.loc[algo, mag] > 0 and low_samples.loc[algo, mag] < 3:
                        print(f"   {algo} @ {mag}x: n = {low_samples.loc[algo, mag]}")
    
    # Save to CSV
    tables_dir = os.path.join(_SCRIPT_DIR, "SEM_tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    sample_summary = df.groupby('algorithm').agg({
        'file': 'count',
        'porosity_%': 'count',
        'avg_thickness_um': 'count'
    }).rename(columns={'file': 'total_images', 
                       'porosity_%': 'n_porosity', 
                       'avg_thickness_um': 'n_thickness'})
    
    sample_summary.to_csv(os.path.join(tables_dir, "sample_sizes.csv"))
    print(f"\n✔ Sample size report saved → {tables_dir}/sample_sizes.csv")


# ============================================================
# PER POWER-SPEED FOLDER ANALYSIS
# ============================================================

def analyze_per_power_speed_folder(main_csv_path, base_folder):
    """
    Her güç-hız klasörü için (örn. 20-10, 20-20, 35-20, 80-80) 
    tüm analizleri ayrı ayrı çalıştırır ve sonuçları ilgili klasörlere kaydeder.
    
    Her klasör içinde 'analysis_results' alt klasörü oluşturulur:
    - CSV sonuçları
    - Grafikler (figures/)
    - İstatistik raporları
    - LaTeX tabloları
    
    Args:
        main_csv_path: Ana SEM_results_full.csv dosyasının yolu
        base_folder: Proje ana klasörü
    """
    from statistical_analysis import (
        analyze_metric_by_algorithm,
        generate_descriptive_table,
        generate_anova_table,
        save_statistical_report,
    )
    
    df_full = pd.read_csv(main_csv_path)
    
    if 'power' not in df_full.columns or 'speed' not in df_full.columns:
        print("⚠️ Power/Speed columns not found in CSV")
        return
    
    # Tüm güç-hız kombinasyonlarını bul
    power_speed_combinations = df_full.groupby(['power', 'speed']).size().reset_index(name='count')
    
    print("\n" + "=" * 80)
    print("PER POWER-SPEED FOLDER ANALYSIS")
    print("=" * 80)
    print(f"\nFound {len(power_speed_combinations)} power-speed combinations:")
    for _, row in power_speed_combinations.iterrows():
        print(f"  {row['power']}-{row['speed']}: {row['count']} samples")
    
    # Her kombinasyon için analiz yap
    for _, row in power_speed_combinations.iterrows():
        power = int(row['power'])
        speed = int(row['speed'])
        folder_name = f"{power}-{speed}"
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {folder_name}")
        print(f"{'='*80}")
        
        # Güç-hız klasörünün yolu
        ps_folder = os.path.join(base_folder, folder_name)
        if not os.path.isdir(ps_folder):
            print(f"⚠️ Folder not found: {ps_folder}")
            continue
        
        # Alt veri setini filtrele
        df_ps = df_full[(df_full['power'] == power) & (df_full['speed'] == speed)].copy()
        
        if df_ps.empty:
            print(f"⚠️ No data for {folder_name}")
            continue
        
        print(f"  Total samples: n = {len(df_ps)}")
        
        # Sonuçlar için klasör oluştur
        results_folder = os.path.join(ps_folder, "analysis_results")
        figures_folder = os.path.join(results_folder, "figures")
        tables_folder = os.path.join(results_folder, "tables")
        
        os.makedirs(results_folder, exist_ok=True)
        os.makedirs(figures_folder, exist_ok=True)
        os.makedirs(tables_folder, exist_ok=True)
        
        # 1) CSV kaydet
        csv_path = os.path.join(results_folder, f"SEM_results_{folder_name}.csv")
        df_ps.to_csv(csv_path, index=False)
        print(f"  ✔ CSV saved → {csv_path}")
        
        # 2) Algoritma başına örnek sayısı
        print(f"\n  Samples per algorithm:")
        for alg in df_ps["algorithm"].unique():
            n = len(df_ps[df_ps["algorithm"] == alg])
            print(f"    {alg}: n = {n}")
        
        # 3) İstatistiksel analiz
        metrics = [
            "porosity_%",
            "avg_pore_area_um2",
            "avg_pore_equiv_diam_um",
            "avg_thickness_um",
            "connectivity_index",
            "fractal_dimension",
            "contrast",
            "correlation",
            "energy",
            "homogeneity",
            "coefficient_of_variation",
            "regional_cv",
            "intensity_entropy",
            "uniformity_index",
        ]
        
        print(f"\n  Running statistical analysis...")
        results_dict = {}
        
        for metric in metrics:
            if metric not in df_ps.columns:
                continue
            
            # En az 2 farklı algoritma ve yeterli veri var mı kontrol et
            metric_data = df_ps[["algorithm", metric]].dropna()
            if metric_data.empty or len(metric_data["algorithm"].unique()) < 2:
                continue
            
            try:
                results = analyze_metric_by_algorithm(df_ps, metric, algorithm_col='algorithm')
                if 'comparison' in results:
                    results_dict[metric] = results
            except Exception as e:
                print(f"  ⚠️ Could not analyze {metric}: {e}")
        
        # 4) İstatistik raporunu kaydet
        if results_dict:
            report_path = os.path.join(results_folder, f"statistical_report_{folder_name}.txt")
            save_statistical_report(results_dict, report_path)
            
            # LaTeX tabloları
            desc_table = generate_descriptive_table(results_dict, latex=True)
            with open(os.path.join(tables_folder, f"descriptive_stats_{folder_name}.tex"), 'w') as f:
                f.write(desc_table)
            
            anova_table = generate_anova_table(results_dict, latex=True)
            with open(os.path.join(tables_folder, f"anova_results_{folder_name}.tex"), 'w') as f:
                f.write(anova_table)
            
            print(f"  ✔ Statistical report saved")
            print(f"  ✔ LaTeX tables saved")
        
        # 5) Grafikler oluştur
        _generate_ps_figures(df_ps, figures_folder, folder_name)
        
        print(f"\n  ✔ Analysis complete for {folder_name}")
        print(f"    Results saved in: {results_folder}")
    
    print("\n" + "=" * 80)
    print("PER POWER-SPEED ANALYSIS COMPLETE!")
    print("=" * 80)


def _generate_ps_figures(df_ps, figures_folder, folder_name):
    """
    Tek bir güç-hız kombinasyonu için tüm grafikleri oluşturur.
    
    Args:
        df_ps: Filtrelenmiş DataFrame (tek güç-hız kombinasyonu)
        figures_folder: Grafiklerin kaydedileceği klasör
        folder_name: Güç-hız adı (örn. "20-10")
    """
    
    # Metrik konfigürasyonu
    metric_config = {
        # Porosity metrics
        "porosity_%": {"unit": "%", "label": "Porosity"},
        "avg_pore_area_um2": {"unit": "µm²", "label": "Average Pore Area"},
        "avg_pore_equiv_diam_um": {"unit": "µm", "label": "Average Pore Diameter"},
        
        # Fine structure metrics
        "avg_thickness_um": {"unit": "µm", "label": "Average Fiber Thickness"},
        "connectivity_index": {"unit": "", "label": "Connectivity Index"},
        "fractal_dimension": {"unit": "", "label": "Fractal Dimension"},
        
        # Homogeneity metrics
        "coefficient_of_variation": {"unit": "%", "label": "Coefficient of Variation"},
        "regional_cv": {"unit": "%", "label": "Regional CV"},
        "intensity_entropy": {"unit": "", "label": "Intensity Entropy"},
        "local_variance_std": {"unit": "", "label": "Local Variance Std"},
        "uniformity_index": {"unit": "", "label": "Uniformity Index"},
        
        # GLCM texture metrics
        "contrast": {"unit": "", "label": "GLCM Contrast"},
        "correlation": {"unit": "", "label": "GLCM Correlation"},
        "energy": {"unit": "", "label": "GLCM Energy"},
        "homogeneity": {"unit": "", "label": "GLCM Homogeneity"},
    }
    
    alg_order = ALGORITHM_ORDER
    generated_count = 0
    
    # --- 1) Her metrik için bar chart ---
    for metric, config in metric_config.items():
        if metric not in df_ps.columns:
            continue
        
        metric_data = df_ps[["algorithm", metric]].dropna().copy()
        
        if metric_data.empty or len(metric_data) < 2:
            continue
        
        # Algoritma isimlerini standartlaştır
        metric_data["algorithm"] = (
            metric_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        # İstatistikleri hesapla
        stats = metric_data.groupby("algorithm")[metric].agg(["mean", "std", "count"]).reset_index()
        stats["std"] = stats["std"].fillna(0)
        
        # Mevcut algoritmaları filtrele
        available_algs = [a for a in alg_order if a in stats["algorithm"].values]
        if len(available_algs) < 2:
            continue
        
        stats = stats.set_index("algorithm").loc[available_algs].reset_index()
        
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=(6, 5))
        
        x = np.arange(len(available_algs))
        bar_colors = [ALGORITHM_COLORS.get(a, '#4a4a4a') for a in available_algs]
        
        # Bar chart
        bars = ax.bar(x, stats["mean"], yerr=stats["std"],
                     capsize=6, color=bar_colors, edgecolor='black',
                     linewidth=1.5, alpha=0.9,
                     error_kw={'linewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})
        
        # Değer etiketleri
        for i, (mean, std) in enumerate(zip(stats["mean"], stats["std"])):
            y_pos = mean + std + 0.05 * (stats["mean"] + stats["std"]).max()
            if abs(mean) < 0.001 and mean != 0:
                label = f"{mean:.1e}"
            else:
                label = f"{mean:.2f}±{std:.2f}"
            ax.text(i, y_pos, label, ha='center', va='bottom',
                   fontweight='bold', fontsize=FONT_SIZE - 1)
        
        # Eksen ayarları
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold')
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_xlabel("Algorithm", fontweight='bold')
        ax.set_title(f"{config['label']} ({folder_name})", fontweight='bold')
        
        # Y ekseni ayarları
        y_max = (stats["mean"] + stats["std"]).max()
        ax.set_ylim(0, y_max * 1.35)
        
        # Stil
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        plt.tight_layout()
        
        # Kaydet
        safe_metric = metric.replace("%", "pct").replace("/", "_")
        fig_path = os.path.join(figures_folder, f"{safe_metric}_{folder_name}.png")
        plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_count += 1
    
    # --- 2) Box plot comparison ---
    _generate_ps_boxplot(df_ps, figures_folder, folder_name, metric_config, alg_order)
    
    # --- 3) Correlation matrix ---
    _generate_ps_correlation(df_ps, figures_folder, folder_name)
    
    # --- 4) PCA plot ---
    _generate_ps_pca(df_ps, figures_folder, folder_name)
    
    # --- 5) Magnification comparison ---
    _generate_ps_magnification_comparison(df_ps, figures_folder, folder_name, metric_config)
    
    print(f"  ✔ {generated_count} metric figures generated")


def _generate_ps_boxplot(df_ps, figures_folder, folder_name, metric_config, alg_order):
    """Box plot comparison for key metrics."""
    key_metrics = ["porosity_%", "avg_thickness_um", "connectivity_index", "fractal_dimension"]
    available_metrics = [m for m in key_metrics if m in df_ps.columns]
    
    if len(available_metrics) < 2:
        return
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        plot_data = df_ps[["algorithm", metric]].dropna().copy()
        if plot_data.empty:
            ax.set_visible(False)
            continue
        
        plot_data["algorithm"] = (
            plot_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        available_algs = [a for a in alg_order if a in plot_data["algorithm"].values]
        if not available_algs:
            ax.set_visible(False)
            continue
        
        colors = [ALGORITHM_COLORS.get(a, '#4a4a4a') for a in available_algs]
        
        # Box plot
        bp = ax.boxplot(
            [plot_data[plot_data["algorithm"] == alg][metric].values for alg in available_algs],
            patch_artist=True,
            labels=available_algs
        )
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        config = metric_config.get(metric, {"unit": "", "label": metric})
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_title(f"{config['label']}", fontweight='bold')
        
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
    
    # Boş eksenleri gizle
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"Metric Comparison ({folder_name})", fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_folder, f"boxplot_comparison_{folder_name}.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_ps_correlation(df_ps, figures_folder, folder_name):
    """Correlation matrix for metrics."""
    from scipy.stats import spearmanr
    
    metrics = [
        "porosity_%", "avg_pore_area_um2", "avg_thickness_um",
        "connectivity_index", "fractal_dimension",
        "contrast", "correlation", "energy", "homogeneity"
    ]
    
    available = [m for m in metrics if m in df_ps.columns]
    data = df_ps[available].dropna()
    
    if data.empty or len(available) < 3:
        return
    
    # Spearman korelasyon
    corr_matrix = data.corr(method='spearman')
    
    # Grafik
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Etiketleri kısalt
    short_labels = [prettify_metric_name(m).split('(')[0].strip()[:15] for m in available]
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Spearman ρ'},
        ax=ax,
        xticklabels=short_labels,
        yticklabels=short_labels
    )
    
    ax.set_title(f"Correlation Matrix ({folder_name})", fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    
    plt.tight_layout()
    
    fig_path = os.path.join(figures_folder, f"correlation_matrix_{folder_name}.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_ps_pca(df_ps, figures_folder, folder_name):
    """PCA visualization for the power-speed subset."""
    numeric_cols = df_ps.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['power', 'speed', 'magnification']]
    
    if len(feature_cols) < 3:
        return
    
    X = df_ps[feature_cols].dropna()
    if len(X) < 5:
        return
    
    # İndeksleri hizala
    valid_idx = X.index
    X = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    if 'algorithm' in df_ps.columns:
        algorithms = df_ps.loc[valid_idx, 'algorithm'].unique()
        for alg in algorithms:
            mask = df_ps.loc[valid_idx, 'algorithm'] == alg
            display_name = ALGORITHM_NAME_MAP.get(alg, alg).replace("_", " ").title()
            color = get_algorithm_color(alg)
            ax.scatter(Xp[mask, 0], Xp[mask, 1],
                      c=color, label=display_name,
                      s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.legend(loc='best', framealpha=0.9)
    
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontweight="bold")
    ax.set_title(f"PCA Analysis ({folder_name})", fontweight="bold")
    
    apply_axis_style(ax)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_folder, f"pca_{folder_name}.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_ps_magnification_comparison(df_ps, figures_folder, folder_name, metric_config):
    """Magnification comparison for key metrics."""
    if 'magnification' not in df_ps.columns:
        return
    
    key_metrics = ["porosity_%", "avg_thickness_um", "fractal_dimension"]
    available_metrics = [m for m in key_metrics if m in df_ps.columns]
    
    if not available_metrics:
        return
    
    mags = sorted(df_ps['magnification'].unique())
    if len(mags) < 2:
        return
    
    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for alg in df_ps['algorithm'].unique():
            alg_data = df_ps[df_ps['algorithm'] == alg]
            
            means = []
            stds = []
            valid_mags = []
            
            for mag in mags:
                mag_data = alg_data[alg_data['magnification'] == mag][metric].dropna()
                if len(mag_data) > 0:
                    means.append(mag_data.mean())
                    stds.append(mag_data.std())
                    valid_mags.append(mag)
            
            if valid_mags:
                display_name = ALGORITHM_NAME_MAP.get(alg, alg).replace("_", " ").title()
                color = get_algorithm_color(alg)
                ax.errorbar(valid_mags, means, yerr=stds,
                           marker='o', label=display_name, color=color,
                           linewidth=2, markersize=8, capsize=4)
        
        config = metric_config.get(metric, {"unit": "", "label": metric})
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_xlabel("Magnification (×)", fontweight='bold')
        ax.set_title(f"{config['label']} vs Magnification ({folder_name})", fontweight='bold')
        ax.set_xscale('log')
        ax.legend(loc='best', framealpha=0.9)
        
        apply_axis_style(ax)
        plt.tight_layout()
        
        safe_metric = metric.replace("%", "pct").replace("/", "_")
        fig_path = os.path.join(figures_folder, f"mag_comparison_{safe_metric}_{folder_name}.png")
        plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()


# ============================================================
# GROUPED POWER-SPEED ANALYSIS (Combined Analysis)
# ============================================================

def analyze_grouped_power_speed(main_csv_path, base_folder):
    """
    Güç-hız kombinasyonlarını gruplar halinde analiz eder:
    - Grup 1: 20-10 ve 35-20 (düşük-orta güç)
    - Grup 2: 20-20 ve 80-80 (yüksek güç veya yüksek hız)
    
    Her grup için:
    - Karşılaştırmalı grafikler (grouped bar charts)
    - İstatistiksel analizler
    - Korelasyon matrisleri
    
    Args:
        main_csv_path: Ana SEM_results_full.csv dosyasının yolu
        base_folder: Proje ana klasörü
    """
    from statistical_analysis import (
        analyze_metric_by_algorithm,
        generate_descriptive_table,
        generate_anova_table,
        save_statistical_report,
    )
    
    df_full = pd.read_csv(main_csv_path)
    
    if 'power' not in df_full.columns or 'speed' not in df_full.columns:
        print("⚠️ Power/Speed columns not found in CSV")
        return
    
    # Grupları tanımla (C1=20-10, C2=20-20, C3=35-20, C4=80-80)
    groups = {
        "Group1_C1_C3": [(20, 10), (35, 20)],   # C1 & C3
        "Group2_C2_C4": [(20, 20), (80, 80)],   # C2 & C4
    }
    
    print("\n" + "=" * 80)
    print("GROUPED POWER-SPEED ANALYSIS")
    print("=" * 80)
    print("\nCondition mapping: C1=20-10, C2=20-20, C3=35-20, C4=80-80")
    print("\nGroup definitions:")
    for group_name, ps_list in groups.items():
        ps_str = " & ".join([PS_NAME_MAP.get(f"{p}-{s}", f"{p}-{s}") for p, s in ps_list])
        print(f"  {group_name}: {ps_str}")
    
    # Her grup için analiz yap
    for group_name, ps_combinations in groups.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING: {group_name}")
        print(f"{'='*80}")
        
        # Alt veri setini filtrele
        mask = pd.Series([False] * len(df_full))
        for power, speed in ps_combinations:
            mask |= ((df_full['power'] == power) & (df_full['speed'] == speed))
        
        df_group = df_full[mask].copy()
        
        if df_group.empty:
            print(f"⚠️ No data for {group_name}")
            continue
        
        # Power-speed label oluştur ve C formatına dönüştür
        df_group['power_speed'] = df_group['power'].astype(str) + '-' + df_group['speed'].astype(str)
        df_group['condition'] = df_group['power_speed'].map(PS_NAME_MAP)
        ps_labels = [f"{p}-{s}" for p, s in ps_combinations]
        
        # Metrik konfigürasyonu
        metric_config = {
            "porosity_%": {"unit": "%", "label": "Porosity"},
            "avg_pore_area_um2": {"unit": "µm²", "label": "Average Pore Area"},
            "avg_pore_equiv_diam_um": {"unit": "µm", "label": "Average Pore Diameter"},
            "avg_thickness_um": {"unit": "µm", "label": "Average Fiber Thickness"},
            "connectivity_index": {"unit": "", "label": "Connectivity Index"},
            "fractal_dimension": {"unit": "", "label": "Fractal Dimension"},
            "coefficient_of_variation": {"unit": "%", "label": "Coefficient of Variation"},
            "regional_cv": {"unit": "%", "label": "Regional CV"},
            "intensity_entropy": {"unit": "", "label": "Intensity Entropy"},
            "local_variance_std": {"unit": "", "label": "Local Variance Std"},
            "uniformity_index": {"unit": "", "label": "Uniformity Index"},
            "contrast": {"unit": "", "label": "GLCM Contrast"},
            "correlation": {"unit": "", "label": "GLCM Correlation"},
            "energy": {"unit": "", "label": "GLCM Energy"},
            "homogeneity": {"unit": "", "label": "GLCM Homogeneity"},
        }
        
        print(f"  Total samples: n = {len(df_group)}")
        
        # Sonuçlar için klasör oluştur
        results_folder = os.path.join(base_folder, f"SEM_grouped_analysis", group_name)
        figures_folder = os.path.join(results_folder, "figures")
        tables_folder = os.path.join(results_folder, "tables")
        
        os.makedirs(results_folder, exist_ok=True)
        os.makedirs(figures_folder, exist_ok=True)
        os.makedirs(tables_folder, exist_ok=True)
        
        # 1) CSV kaydet
        csv_path = os.path.join(results_folder, f"SEM_results_{group_name}.csv")
        df_group.to_csv(csv_path, index=False)
        print(f"  ✔ CSV saved → {csv_path}")
        
        # 2) Örneklem dağılımını göster
        print(f"\n  Samples per condition and algorithm:")
        for ps in df_group['power_speed'].unique():
            condition = PS_NAME_MAP.get(ps, ps)
            print(f"    {condition} ({ps}):")
            ps_data = df_group[df_group['power_speed'] == ps]
            for alg in ps_data['algorithm'].unique():
                n = len(ps_data[ps_data['algorithm'] == alg])
                print(f"      {alg}: n = {n}")
        
        # 3) İstatistiksel analiz
        metrics = [
            "porosity_%",
            "avg_pore_area_um2",
            "avg_pore_equiv_diam_um",
            "avg_thickness_um",
            "connectivity_index",
            "fractal_dimension",
            "contrast",
            "correlation",
            "energy",
            "homogeneity",
            "coefficient_of_variation",
            "regional_cv",
            "intensity_entropy",
            "uniformity_index",
        ]
        
        print(f"\n  Running statistical analysis...")
        results_dict = {}
        
        for metric in metrics:
            if metric not in df_group.columns:
                continue
            
            metric_data = df_group[["algorithm", metric]].dropna()
            if metric_data.empty or len(metric_data["algorithm"].unique()) < 2:
                continue
            
            try:
                results = analyze_metric_by_algorithm(df_group, metric, algorithm_col='algorithm')
                if 'comparison' in results:
                    results_dict[metric] = results
            except Exception as e:
                print(f"  ⚠️ Could not analyze {metric}: {e}")
        
        # 4) İstatistik raporunu kaydet
        if results_dict:
            report_path = os.path.join(results_folder, f"statistical_report_{group_name}.txt")
            save_statistical_report(results_dict, report_path)
            
            # LaTeX tabloları
            try:
                desc_table = generate_descriptive_table(results_dict, latex=True)
                with open(os.path.join(tables_folder, f"descriptive_stats_{group_name}.tex"), 'w') as f:
                    f.write(desc_table)
                
                anova_table = generate_anova_table(results_dict, latex=True)
                with open(os.path.join(tables_folder, f"anova_results_{group_name}.tex"), 'w') as f:
                    f.write(anova_table)
                
                print(f"  ✔ Statistical report and LaTeX tables saved")
            except Exception as e:
                print(f"  ⚠️ Could not generate tables: {e}")
        
        # 5) Gruplandırılmış grafikler oluştur
        _generate_grouped_figures(df_group, figures_folder, group_name, ps_combinations)
        
        # 6) Excel dosyası oluştur (veriler + grafikler)
        excel_path = os.path.join(results_folder, f"SEM_results_{group_name}.xlsx")
        _save_results_to_excel(df_group, figures_folder, excel_path, group_name, ps_labels, metric_config)
        print(f"  ✔ Excel with charts saved → {excel_path}")
        
        print(f"\n  ✔ Analysis complete for {group_name}")
        print(f"    Results saved in: {results_folder}")
    
    print("\n" + "=" * 80)
    print("GROUPED POWER-SPEED ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n✔ Results saved in: {os.path.join(base_folder, 'SEM_grouped_analysis')}/")


def _save_results_to_excel(df_group, figures_folder, excel_path, group_name, ps_labels, metric_config):
    """
    Sonuçları ve grafikleri Excel dosyasına kaydeder.
    
    Args:
        df_group: Grup verisi DataFrame
        figures_folder: Grafiklerin bulunduğu klasör
        excel_path: Kaydedilecek Excel dosya yolu
        group_name: Grup adı
        ps_labels: Power-speed etiketleri listesi
        metric_config: Metrik yapılandırması
    """
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.chart import BarChart, Reference
    from openpyxl.chart.label import DataLabelList
    from openpyxl.chart.series import DataPoint
    from openpyxl.drawing.fill import PatternFillProperties, ColorChoice
    
    wb = Workbook()
    
    # Renk paleti (hex without #)
    COLORS = {
        "C1": "2C3E50",  # Koyu lacivert
        "C2": "6C3483",  # Mor
        "C3": "1A5276",  # Koyu mavi
        "C4": "922B21",  # Koyu kırmızı
    }
    ALG_COLORS = {
        "Stucki": "1B4F72",
        "Floyd-Steinberg": "2E4053",
        "Jarvis": "4A235A",
        "Standard": "17202A",
    }
    
    # =========== 1) RAW DATA SHEET ===========
    ws_data = wb.active
    ws_data.title = "Raw Data"
    
    # DataFrame'i Excel'e yaz
    for r_idx, row in enumerate(dataframe_to_rows(df_group, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws_data.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 1:  # Header
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="2C3E50", end_color="2C3E50", fill_type="solid")
                cell.alignment = Alignment(horizontal="center")
    
    # Sütun genişliklerini ayarla
    for col in ws_data.columns:
        max_length = max(len(str(cell.value or "")) for cell in col)
        ws_data.column_dimensions[col[0].column_letter].width = min(max_length + 2, 20)
    
    # =========== 2) SUMMARY STATISTICS SHEET ===========
    ws_stats = wb.create_sheet("Summary Statistics")
    
    key_metrics = ["porosity_%", "avg_pore_area_um2", "avg_thickness_um", 
                   "connectivity_index", "fractal_dimension", "contrast"]
    available_metrics = [m for m in key_metrics if m in df_group.columns]
    
    # Header
    headers = ["Condition", "Algorithm", "n"] + [metric_config.get(m, {"label": m})["label"] for m in available_metrics]
    for c_idx, header in enumerate(headers, 1):
        cell = ws_stats.cell(row=1, column=c_idx, value=header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="1A5276", end_color="1A5276", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")
    
    # Verileri yaz
    row_idx = 2
    for ps in ps_labels:
        condition = PS_NAME_MAP.get(ps, ps)
        ps_data = df_group[df_group['power_speed'] == ps]
        
        for alg in df_group['algorithm'].unique():
            alg_data = ps_data[ps_data['algorithm'] == alg]
            if alg_data.empty:
                continue
            
            alg_display = ALGORITHM_NAME_MAP.get(alg, alg).replace("_", " ").title()
            
            ws_stats.cell(row=row_idx, column=1, value=condition)
            ws_stats.cell(row=row_idx, column=2, value=alg_display)
            ws_stats.cell(row=row_idx, column=3, value=len(alg_data))
            
            for m_idx, metric in enumerate(available_metrics, 4):
                mean_val = alg_data[metric].mean()
                std_val = alg_data[metric].std()
                cell = ws_stats.cell(row=row_idx, column=m_idx, value=f"{mean_val:.2f} ± {std_val:.2f}")
                cell.alignment = Alignment(horizontal="center")
            
            row_idx += 1
    
    for col in ws_stats.columns:
        ws_stats.column_dimensions[col[0].column_letter].width = 18
    
    # =========== 3) CHARTS BY CONDITION SHEET ===========
    ws_cond = wb.create_sheet("Charts by Condition")
    
    # Her metrik için Condition bazlı grafik (X: Algorithm, Series: Condition)
    chart_row = 1
    for m_idx, metric in enumerate(available_metrics):
        config = metric_config.get(metric, {"label": metric, "unit": ""})
        
        # Veri tablosu başlangıcı
        data_col = 1
        data_row = chart_row
        
        # Başlık
        ws_cond.cell(row=data_row, column=data_col, value=config["label"]).font = Font(bold=True, size=11)
        data_row += 1
        
        # Header: Algorithm isimleri
        ws_cond.cell(row=data_row, column=data_col, value="")
        alg_list = list(ALGORITHM_ORDER)
        for a_idx, alg in enumerate(alg_list, data_col + 1):
            ws_cond.cell(row=data_row, column=a_idx, value=alg).font = Font(bold=True)
        data_row += 1
        
        # Her condition için satır
        header_row = data_row - 1
        for ps in ps_labels:
            condition = PS_NAME_MAP.get(ps, ps)
            ws_cond.cell(row=data_row, column=data_col, value=condition)
            
            ps_data = df_group[df_group['power_speed'] == ps]
            for a_idx, alg_name in enumerate(alg_list, data_col + 1):
                for alg_key in ALGORITHM_NAME_MAP.keys():
                    if ALGORITHM_NAME_MAP[alg_key].replace("_", " ").title() == alg_name:
                        alg_data = ps_data[ps_data['algorithm'] == alg_key]
                        if not alg_data.empty:
                            ws_cond.cell(row=data_row, column=a_idx, value=round(alg_data[metric].mean(), 3))
                        break
            data_row += 1
        
        # Bar chart oluştur
        chart = BarChart()
        chart.type = "col"
        chart.grouping = "clustered"
        chart.title = config["label"]
        chart.y_axis.title = config.get("unit", "") or "Value"
        chart.x_axis.title = "Algorithm"
        chart.style = 10
        
        # Veri referansları
        data_ref = Reference(ws_cond, min_col=data_col + 1, min_row=header_row, 
                            max_col=data_col + len(alg_list), max_row=data_row - 1)
        cats_ref = Reference(ws_cond, min_col=data_col, min_row=header_row + 1, max_row=data_row - 1)
        
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.shape = 4
        chart.width = 15
        chart.height = 10
        
        # Grafik pozisyonu (sağ tarafa)
        ws_cond.add_chart(chart, f"H{chart_row}")
        
        chart_row = data_row + 2
    
    # =========== 4) CHARTS BY ALGORITHM SHEET ===========
    ws_alg = wb.create_sheet("Charts by Algorithm")
    
    # Her metrik için Algorithm bazlı grafik (X: Condition, Series: Algorithm)
    chart_row = 1
    for m_idx, metric in enumerate(available_metrics):
        config = metric_config.get(metric, {"label": metric, "unit": ""})
        
        data_col = 1
        data_row = chart_row
        
        # Başlık
        ws_alg.cell(row=data_row, column=data_col, value=config["label"]).font = Font(bold=True, size=11)
        data_row += 1
        
        # Header: Condition isimleri
        ws_alg.cell(row=data_row, column=data_col, value="")
        cond_list = [PS_NAME_MAP.get(ps, ps) for ps in ps_labels]
        for c_idx, cond in enumerate(cond_list, data_col + 1):
            ws_alg.cell(row=data_row, column=c_idx, value=cond).font = Font(bold=True)
        data_row += 1
        
        # Her algoritma için satır
        header_row = data_row - 1
        for alg_name in ALGORITHM_ORDER:
            ws_alg.cell(row=data_row, column=data_col, value=alg_name)
            
            for c_idx, ps in enumerate(ps_labels, data_col + 1):
                ps_data = df_group[df_group['power_speed'] == ps]
                for alg_key in ALGORITHM_NAME_MAP.keys():
                    if ALGORITHM_NAME_MAP[alg_key].replace("_", " ").title() == alg_name:
                        alg_data = ps_data[ps_data['algorithm'] == alg_key]
                        if not alg_data.empty:
                            ws_alg.cell(row=data_row, column=c_idx, value=round(alg_data[metric].mean(), 3))
                        break
            data_row += 1
        
        # Bar chart
        chart = BarChart()
        chart.type = "col"
        chart.grouping = "clustered"
        chart.title = config["label"]
        chart.y_axis.title = config.get("unit", "") or "Value"
        chart.x_axis.title = "Condition"
        chart.style = 10
        
        data_ref = Reference(ws_alg, min_col=data_col + 1, min_row=header_row, 
                            max_col=data_col + len(cond_list), max_row=data_row - 1)
        cats_ref = Reference(ws_alg, min_col=data_col, min_row=header_row + 1, max_row=data_row - 1)
        
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.shape = 4
        chart.width = 15
        chart.height = 10
        
        ws_alg.add_chart(chart, f"F{chart_row}")
        
        chart_row = data_row + 2
    
    # =========== 5) MEAN OF CONDITIONS SHEET ===========
    ws_mean = wb.create_sheet("Mean of Conditions")
    
    # Her algoritma için condition ortalamalarının ortalaması
    chart_row = 1
    for m_idx, metric in enumerate(available_metrics):
        config = metric_config.get(metric, {"label": metric, "unit": ""})
        
        data_col = 1
        data_row = chart_row
        
        # Başlık
        ws_mean.cell(row=data_row, column=data_col, value=f"{config['label']} - Mean of Conditions").font = Font(bold=True, size=11)
        data_row += 1
        
        # Header
        ws_mean.cell(row=data_row, column=data_col, value="Algorithm").font = Font(bold=True)
        ws_mean.cell(row=data_row, column=data_col + 1, value="Mean").font = Font(bold=True)
        ws_mean.cell(row=data_row, column=data_col + 2, value="Std").font = Font(bold=True)
        data_row += 1
        
        header_row = data_row - 1
        
        for alg_name in ALGORITHM_ORDER:
            ws_mean.cell(row=data_row, column=data_col, value=alg_name)
            
            # Her condition için ortalama hesapla
            ps_means = []
            for ps in ps_labels:
                ps_data = df_group[df_group['power_speed'] == ps]
                for alg_key in ALGORITHM_NAME_MAP.keys():
                    if ALGORITHM_NAME_MAP[alg_key].replace("_", " ").title() == alg_name:
                        alg_data = ps_data[ps_data['algorithm'] == alg_key]
                        if not alg_data.empty:
                            ps_means.append(alg_data[metric].mean())
                        break
            
            if ps_means:
                mean_of_means = np.mean(ps_means)
                std_of_means = np.std(ps_means) if len(ps_means) > 1 else 0
                ws_mean.cell(row=data_row, column=data_col + 1, value=round(mean_of_means, 3))
                ws_mean.cell(row=data_row, column=data_col + 2, value=round(std_of_means, 3))
            
            data_row += 1
        
        # Bar chart
        chart = BarChart()
        chart.type = "col"
        chart.grouping = "standard"
        chart.title = f"{config['label']} - Mean of ({', '.join([PS_NAME_MAP.get(ps, ps) for ps in ps_labels])})"
        chart.y_axis.title = config.get("unit", "") or "Value"
        chart.x_axis.title = "Algorithm"
        chart.style = 10
        
        data_ref = Reference(ws_mean, min_col=data_col + 1, min_row=header_row, 
                            max_col=data_col + 1, max_row=data_row - 1)
        cats_ref = Reference(ws_mean, min_col=data_col, min_row=header_row + 1, max_row=data_row - 1)
        
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.shape = 4
        chart.width = 12
        chart.height = 8
        
        # Data labels ekle
        chart.dataLabels = DataLabelList()
        chart.dataLabels.showVal = True
        
        ws_mean.add_chart(chart, f"F{chart_row}")
        
        chart_row = data_row + 2
    
    # =========== 6) CONDITION COMPARISON SHEET ===========
    ws_comp = wb.create_sheet("Condition Comparison")
    
    # Her metrik için condition karşılaştırması (tüm algoritmalar birleşik)
    chart_row = 1
    for m_idx, metric in enumerate(available_metrics):
        config = metric_config.get(metric, {"label": metric, "unit": ""})
        
        data_col = 1
        data_row = chart_row
        
        # Başlık
        ws_comp.cell(row=data_row, column=data_col, value=config["label"]).font = Font(bold=True, size=11)
        data_row += 1
        
        # Header
        ws_comp.cell(row=data_row, column=data_col, value="Condition").font = Font(bold=True)
        ws_comp.cell(row=data_row, column=data_col + 1, value="Mean").font = Font(bold=True)
        ws_comp.cell(row=data_row, column=data_col + 2, value="Std").font = Font(bold=True)
        ws_comp.cell(row=data_row, column=data_col + 3, value="Error %").font = Font(bold=True)
        data_row += 1
        
        header_row = data_row - 1
        
        for ps in ps_labels:
            condition = PS_NAME_MAP.get(ps, ps)
            ws_comp.cell(row=data_row, column=data_col, value=condition)
            
            ps_data = df_group[df_group['power_speed'] == ps]
            mean_val = ps_data[metric].mean()
            std_val = ps_data[metric].std()
            pct_error = (std_val / mean_val * 100) if mean_val != 0 else 0
            
            ws_comp.cell(row=data_row, column=data_col + 1, value=round(mean_val, 3))
            ws_comp.cell(row=data_row, column=data_col + 2, value=round(std_val, 3))
            ws_comp.cell(row=data_row, column=data_col + 3, value=f"±{pct_error:.1f}%")
            
            data_row += 1
        
        # Bar chart
        chart = BarChart()
        chart.type = "col"
        chart.grouping = "standard"
        chart.title = f"{config['label']} by Condition"
        chart.y_axis.title = config.get("unit", "") or "Value"
        chart.x_axis.title = "Condition"
        chart.style = 10
        
        data_ref = Reference(ws_comp, min_col=data_col + 1, min_row=header_row, 
                            max_col=data_col + 1, max_row=data_row - 1)
        cats_ref = Reference(ws_comp, min_col=data_col, min_row=header_row + 1, max_row=data_row - 1)
        
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.shape = 4
        chart.width = 10
        chart.height = 8
        
        chart.dataLabels = DataLabelList()
        chart.dataLabels.showVal = True
        
        ws_comp.add_chart(chart, f"G{chart_row}")
        
        chart_row = data_row + 2
    
    # Sütun genişlikleri ayarla
    for ws in [ws_cond, ws_alg, ws_mean, ws_comp]:
        for col_letter in ['A', 'B', 'C', 'D', 'E', 'F']:
            ws.column_dimensions[col_letter].width = 15
    
    # =========== KAYDET ===========
    wb.save(excel_path)


def _generate_grouped_figures(df_group, figures_folder, group_name, ps_combinations):
    """
    Gruplandırılmış veriler için karşılaştırmalı grafikler oluşturur.
    
    Args:
        df_group: Filtrelenmiş DataFrame (grup verisi)
        figures_folder: Grafiklerin kaydedileceği klasör
        group_name: Grup adı
        ps_combinations: [(power, speed), ...] listesi
    """
    
    # Metrik konfigürasyonu
    metric_config = {
        "porosity_%": {"unit": "%", "label": "Porosity"},
        "avg_pore_area_um2": {"unit": "µm²", "label": "Average Pore Area"},
        "avg_pore_equiv_diam_um": {"unit": "µm", "label": "Average Pore Diameter"},
        "avg_thickness_um": {"unit": "µm", "label": "Average Fiber Thickness"},
        "connectivity_index": {"unit": "", "label": "Connectivity Index"},
        "fractal_dimension": {"unit": "", "label": "Fractal Dimension"},
        "coefficient_of_variation": {"unit": "%", "label": "Coefficient of Variation"},
        "regional_cv": {"unit": "%", "label": "Regional CV"},
        "intensity_entropy": {"unit": "", "label": "Intensity Entropy"},
        "local_variance_std": {"unit": "", "label": "Local Variance Std"},
        "uniformity_index": {"unit": "", "label": "Uniformity Index"},
        "contrast": {"unit": "", "label": "GLCM Contrast"},
        "correlation": {"unit": "", "label": "GLCM Correlation"},
        "energy": {"unit": "", "label": "GLCM Energy"},
        "homogeneity": {"unit": "", "label": "GLCM Homogeneity"},
    }
    
    ps_labels = [f"{p}-{s}" for p, s in ps_combinations]
    alg_order = ALGORITHM_ORDER
    generated_count = 0
    
    # C isimlerini kullan
    ps_display_labels = [PS_NAME_MAP.get(ps, ps) for ps in ps_labels]
    
    # --- 1) Her metrik için grouped bar chart ---
    for metric, config in metric_config.items():
        if metric not in df_group.columns:
            continue
        
        metric_data = df_group[["algorithm", "power_speed", metric]].dropna().copy()
        
        if metric_data.empty or len(metric_data) < 2:
            continue
        
        # Algoritma isimlerini standartlaştır
        metric_data["algorithm"] = (
            metric_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        # İstatistikleri hesapla
        stats = metric_data.groupby(["algorithm", "power_speed"])[metric].agg(["mean", "std", "count"]).reset_index()
        stats["std"] = stats["std"].fillna(0)
        
        # Mevcut algoritmaları filtrele
        available_algs = [a for a in alg_order if a in stats["algorithm"].values]
        if len(available_algs) < 2:
            continue
        
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(available_algs))
        n_ps = len(ps_labels)
        width = 0.35
        
        # Maksimum yüksekliği takip et
        all_max_heights = []
        
        # Her power-speed için bar çiz
        for i, ps in enumerate(ps_labels):
            ps_stats = stats[stats["power_speed"] == ps]
            ps_display = PS_NAME_MAP.get(ps, ps)  # C1, C2, etc.
            
            means = []
            stds = []
            for alg in available_algs:
                alg_data = ps_stats[ps_stats["algorithm"] == alg]
                if not alg_data.empty:
                    means.append(alg_data["mean"].values[0])
                    stds.append(alg_data["std"].values[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            # Maksimum yüksekliği kaydet
            all_max_heights.extend([m + s for m, s in zip(means, stds)])
            
            offset = (i - (n_ps - 1) / 2) * width
            color = PS_COLORS_PROFESSIONAL.get(ps, '#4a4a4a')
            
            bars = ax.bar(x + offset, means, width, yerr=stds,
                         label=ps_display, color=color, edgecolor='black',
                         linewidth=1.2, alpha=0.9,
                         capsize=4, error_kw={'linewidth': 1.2, 'capthick': 1.2, 'ecolor': 'black'})
            
            # Yüzde hata etiketleri ekle
            for j, (mean_val, std_val) in enumerate(zip(means, stds)):
                if mean_val != 0:
                    pct_error = (std_val / mean_val) * 100
                    y_pos = mean_val + std_val + 0.02 * max(means)
                    ax.text(x[j] + offset, y_pos, f'±{pct_error:.1f}%', 
                           ha='center', va='bottom', fontsize=7, fontweight='bold',
                           color=color)
        
        # Eksen ayarları
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold')
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_xlabel("Algorithm", fontweight='bold')
        ax.set_title(f"{config['label']} - {group_name.replace('_', ' ')}", fontweight='bold', pad=15)
        
        # Y ekseni ayarları - etiketler için yer bırak
        max_height = max(all_max_heights) if all_max_heights else 1
        ax.set_ylim(bottom=0, top=max_height * 1.2)
        
        # Legend
        ax.legend(title="Condition", loc='upper right', framealpha=0.9)
        
        # Stil
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        plt.tight_layout()
        
        # Kaydet
        safe_metric = metric.replace("%", "pct").replace("/", "_")
        fig_path = os.path.join(figures_folder, f"grouped_{safe_metric}.png")
        plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        generated_count += 1
    
    # --- 2) Condition karşılaştırma (ortalamalar) ---
    _generate_ps_comparison_figure(df_group, figures_folder, group_name, ps_labels, metric_config)
    
    # --- 3) Box plot karşılaştırma ---
    _generate_grouped_boxplot(df_group, figures_folder, group_name, ps_labels, metric_config, alg_order)
    
    # --- 4) Heatmap karşılaştırma ---
    _generate_grouped_heatmap(df_group, figures_folder, group_name, ps_labels, metric_config, alg_order)
    
    # --- 5) Korelasyon matrisi ---
    _generate_grouped_correlation(df_group, figures_folder, group_name)
    
    # --- 6) PCA ---
    _generate_grouped_pca(df_group, figures_folder, group_name)
    
    # --- 7) GRUP ORTALAMA GRAFİKLERİ (YENİ) ---
    _generate_group_mean_figures(df_group, figures_folder, group_name, ps_labels, metric_config)
    
    # --- 8) MAKALE İÇİN BİRLEŞİK GRAFİK ---
    _generate_paper_figure(df_group, figures_folder, group_name, ps_labels, metric_config, alg_order)
    
    print(f"  ✔ {generated_count} grouped metric figures generated")


def _generate_paper_figure(df_group, figures_folder, group_name, ps_labels, metric_config, alg_order):
    """
    Makale için 4 ana metriği tek satırda yan yana birleştirir.
    a) Porosity, b) Average Pore Area, c) Average Fiber Thickness, d) Connectivity Index
    Her grafik: algoritmaların condition ortalamalarının ortalamasını gösterir.
    """
    
    # 4 ana metrik
    paper_metrics = [
        ("porosity_%", "a)"),
        ("avg_pore_area_um2", "b)"),
        ("avg_thickness_um", "c)"),
        ("connectivity_index", "d)"),
    ]
    
    # Sadece mevcut metrikleri al
    available = [(m, label) for m, label in paper_metrics if m in df_group.columns]
    if len(available) < 4:
        return
    
    # Daha küçük figür boyutu, fontlar orantılı olarak daha büyük görünür
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Algoritma isimlerini standartlaştır
    df_plot = df_group.copy()
    df_plot["algorithm"] = (
        df_plot["algorithm"].replace(ALGORITHM_NAME_MAP)
        .astype(str).str.replace("_", " ").str.title()
    )
    
    available_algs = [a for a in alg_order if a in df_plot["algorithm"].values]
    
    # Profesyonel algoritma renkleri
    ALG_COLORS = {
        "Stucki": "#1B4F72",
        "Floyd-Steinberg": "#2E4053",
        "Jarvis": "#4A235A",
        "Standard": "#17202A",
    }
    
    for idx, (metric, panel_label) in enumerate(available):
        ax = axes[idx]
        config = metric_config.get(metric, {"unit": "", "label": metric})
        
        x = np.arange(len(available_algs))
        
        means = []
        stds = []
        colors = []
        
        for alg in available_algs:
            # Her condition için bu algoritmanın ortalamasını hesapla
            ps_means = []
            for ps in ps_labels:
                subset = df_plot[(df_plot["algorithm"] == alg) & (df_plot["power_speed"] == ps)][metric].dropna()
                if len(subset) > 0:
                    ps_means.append(subset.mean())
            
            # Condition ortalamalarının ortalaması
            if len(ps_means) > 0:
                mean_of_means = np.mean(ps_means)
                std_of_means = np.std(ps_means) if len(ps_means) > 1 else 0
            else:
                mean_of_means = 0
                std_of_means = 0
            
            means.append(mean_of_means)
            stds.append(std_of_means)
            colors.append(ALG_COLORS.get(alg, '#4a4a4a'))
        
        # Bar chart
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
                     linewidth=1.2, alpha=0.9,
                     capsize=5, error_kw={'linewidth': 1.2, 'capthick': 1.2, 'ecolor': 'black'})
        
        # Değer ve yüzde hata etiketleri (2 satır: değer üstte, hata altta)
        for i, (mean_val, std_val) in enumerate(zip(means, stds)):
            if mean_val != 0:
                pct_error = (std_val / mean_val) * 100 if mean_val != 0 else 0
                y_pos = mean_val + std_val + 0.03 * max(means) if max(means) > 0 else 0.1
                
                # Değer formatı
                if abs(mean_val) >= 100:
                    val_str = f"{mean_val:.0f}"
                elif abs(mean_val) >= 10:
                    val_str = f"{mean_val:.1f}"
                else:
                    val_str = f"{mean_val:.2f}"
                
                # İki satırlı etiket: değer ve ±hata%
                ax.text(i, y_pos, f'{val_str}\n±{pct_error:.1f}%', 
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Eksen ayarları - BÜYÜK FONTLAR
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold', fontsize=11, rotation=45, ha='right')
        ax.tick_params(axis='y', labelsize=11)
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold', fontsize=14)
        
        # Başlık olarak sadece metrik adı - BÜYÜK FONT
        ax.set_title(config['label'], fontweight='bold', fontsize=14, pad=12)
        
        # Y ekseni ayarları - 2 satır etiket için daha fazla alan
        max_height = max([m + s for m, s in zip(means, stds)]) if means else 1
        ax.set_ylim(bottom=0, top=max_height * 1.4)
        
        # a), b), c), d) etiketi sol üst köşeye - BÜYÜK FONT
        ax.text(-0.08, 1.02, panel_label, transform=ax.transAxes, 
               fontsize=16, fontweight='bold', va='bottom', ha='left')
        
        # Stil
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
    
    # Üst başlık - BÜYÜK FONT
    cond_str = " & ".join([PS_NAME_MAP.get(ps, ps) for ps in ps_labels])
    fig.suptitle(f"Mean of Conditions ({cond_str}) - {group_name.replace('_', ' ')}", 
                fontweight='bold', fontsize=16, y=1.05)
    
    plt.tight_layout()
    
    # Kaydet - yüksek DPI
    fig_path = os.path.join(figures_folder, f"paper_figure_combined.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✔ Paper figure saved → {fig_path}")


# Profesyonel renk paleti (global)
PS_COLORS_PROFESSIONAL = {
    "20-10": "#2C3E50",  # Koyu lacivert - C1
    "35-20": "#1A5276",  # Koyu mavi - C3
    "20-20": "#6C3483",  # Mor - C2
    "80-80": "#922B21",  # Koyu kırmızı - C4
    # C isimleri için de aynı renkler
    "C1": "#2C3E50",
    "C2": "#6C3483",
    "C3": "#1A5276",
    "C4": "#922B21",
}

# Power-Speed → C isim dönüşümü
PS_NAME_MAP = {
    "20-10": "C1",
    "20-20": "C2",
    "35-20": "C3",
    "80-80": "C4",
}

def get_ps_display_name(ps_name):
    """Power-speed adını C formatına dönüştürür."""
    return PS_NAME_MAP.get(ps_name, ps_name)


def _generate_group_mean_figures(df_group, figures_folder, group_name, ps_labels, metric_config):
    """
    Her algoritma için gruptaki condition ortalamalarının ortalamasını gösterir.
    
    Örnek: Grup 1 (C1 & C3) için Stucki:
    - C1'deki Stucki ortalaması = X1
    - C3'deki Stucki ortalaması = X2
    - Grafikteki değer = (X1 + X2) / 2
    """
    
    key_metrics = ["porosity_%", "avg_pore_area_um2", "avg_thickness_um", 
                   "connectivity_index", "fractal_dimension", "contrast"]
    available_metrics = [m for m in key_metrics if m in df_group.columns]
    
    if len(available_metrics) < 2:
        return
    
    # Standardize algorithm names
    df_plot = df_group.copy()
    df_plot["algorithm"] = (
        df_plot["algorithm"].replace(ALGORITHM_NAME_MAP)
        .astype(str).str.replace("_", " ").str.title()
    )
    
    alg_order = ALGORITHM_ORDER
    available_algs = [a for a in alg_order if a in df_plot["algorithm"].values]
    
    if len(available_algs) < 2:
        return
    
    # Profesyonel algoritma renkleri
    ALG_COLORS_PROFESSIONAL = {
        "Stucki": "#1B4F72",
        "Floyd-Steinberg": "#2E4053",
        "Jarvis": "#4A235A",
        "Standard": "#17202A",
    }
    
    # C display isimlerini al
    ps_display_labels = [PS_NAME_MAP.get(ps, ps) for ps in ps_labels]
    
    # Her metrik için ayrı grafik oluştur
    for metric in available_metrics:
        config = metric_config.get(metric, {"unit": "", "label": metric})
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(len(available_algs))
        
        means = []
        stds = []
        colors = []
        
        for alg in available_algs:
            # Her condition için bu algoritmanın ortalamasını hesapla
            ps_means = []
            for ps in ps_labels:
                subset = df_plot[(df_plot["algorithm"] == alg) & (df_plot["power_speed"] == ps)][metric].dropna()
                if len(subset) > 0:
                    ps_means.append(subset.mean())
            
            # Condition ortalamalarının ortalaması
            if len(ps_means) > 0:
                mean_of_means = np.mean(ps_means)
                std_of_means = np.std(ps_means) if len(ps_means) > 1 else 0
            else:
                mean_of_means = 0
                std_of_means = 0
            
            means.append(mean_of_means)
            stds.append(std_of_means)
            colors.append(ALG_COLORS_PROFESSIONAL.get(alg, '#4a4a4a'))
        
        # Bar chart
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
                     linewidth=1.2, alpha=0.9,
                     capsize=6, error_kw={'linewidth': 1.2, 'capthick': 1.2, 'ecolor': 'black'})
        
        # Değer ve yüzde hata etiketleri
        for i, (mean_val, std_val) in enumerate(zip(means, stds)):
            if mean_val != 0:
                pct_error = (std_val / mean_val) * 100 if mean_val != 0 else 0
                y_pos = mean_val + std_val + 0.03 * max(means)
                
                if abs(mean_val) < 0.01:
                    val_label = f"{mean_val:.2e}"
                else:
                    val_label = f"{mean_val:.2f}"
                
                ax.text(i, y_pos, f"{val_label}\n±{pct_error:.1f}%", 
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold')
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_xlabel("Algorithm", fontweight='bold')
        
        # Başlık: Hangi condition'ların ortalaması olduğunu belirt (C isimlerini kullan)
        ps_str = " & ".join(ps_display_labels)
        ax.set_title(f"{config['label']} - Mean of ({ps_str})", fontweight='bold', pad=15)
        
        # Y eksenini etiketler için genişlet (üst üste binmeyi önle)
        max_height = max([m + s for m, s in zip(means, stds)]) if means else 1
        ax.set_ylim(bottom=0, top=max_height * 1.25)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        safe_metric = metric.replace("%", "pct").replace("/", "_")
        fig_path = os.path.join(figures_folder, f"mean_of_ps_{safe_metric}.png")
        plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Tüm metrikleri bir arada gösteren multi-panel grafik
    _generate_combined_mean_figure(df_plot, figures_folder, group_name, ps_labels, 
                                   available_metrics, metric_config, available_algs, ALG_COLORS_PROFESSIONAL)


def _generate_combined_mean_figure(df_plot, figures_folder, group_name, ps_labels, 
                                   available_metrics, metric_config, available_algs, ALG_COLORS_PROFESSIONAL):
    """Tüm metriklerin grup ortalamasını tek bir multi-panel grafikte gösterir."""
    
    n_metrics = min(len(available_metrics), 6)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(available_metrics[:6]):
        ax = axes[idx]
        config = metric_config.get(metric, {"unit": "", "label": metric})
        
        x = np.arange(len(available_algs))
        
        means = []
        stds = []
        colors = []
        
        for alg in available_algs:
            ps_means = []
            for ps in ps_labels:
                subset = df_plot[(df_plot["algorithm"] == alg) & (df_plot["power_speed"] == ps)][metric].dropna()
                if len(subset) > 0:
                    ps_means.append(subset.mean())
            
            if len(ps_means) > 0:
                mean_of_means = np.mean(ps_means)
                std_of_means = np.std(ps_means) if len(ps_means) > 1 else 0
            else:
                mean_of_means = 0
                std_of_means = 0
            
            means.append(mean_of_means)
            stds.append(std_of_means)
            colors.append(ALG_COLORS_PROFESSIONAL.get(alg, '#4a4a4a'))
        
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
                     linewidth=1.0, alpha=0.9,
                     capsize=4, error_kw={'linewidth': 1.0, 'capthick': 1.0, 'ecolor': 'black'})
        
        # Yüzde hata etiketleri
        for i, (mean_val, std_val) in enumerate(zip(means, stds)):
            if mean_val != 0:
                pct_error = (std_val / mean_val) * 100
                y_pos = mean_val + std_val + 0.03 * max(means) if max(means) > 0 else 0.1
                ax.text(i, y_pos, f'±{pct_error:.1f}%', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(available_algs, fontweight='bold', fontsize=9)
        
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold', fontsize=10)
        ax.set_title(config['label'], fontweight='bold', pad=10)
        
        # Y eksenini etiketler için genişlet
        max_height = max([m + s for m, s in zip(means, stds)]) if means else 1
        ax.set_ylim(bottom=0, top=max_height * 1.2)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    # Boş panelleri gizle
    for idx in range(len(available_metrics[:6]), len(axes)):
        axes[idx].set_visible(False)
    
    ps_display_str = " & ".join([PS_NAME_MAP.get(ps, ps) for ps in ps_labels])
    fig.suptitle(f"Mean of Conditions ({ps_display_str}) - {group_name.replace('_', ' ')}", 
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_folder, f"combined_mean_of_conditions.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_ps_comparison_figure(df_group, figures_folder, group_name, ps_labels, metric_config):
    """Power-speed kombinasyonları arasında ortalama karşılaştırması."""
    
    key_metrics = ["porosity_%", "avg_thickness_um", "connectivity_index", "fractal_dimension"]
    available_metrics = [m for m in key_metrics if m in df_group.columns]
    
    if len(available_metrics) < 2:
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Her power-speed için ortalama ve std hesapla
        ps_stats = df_group.groupby("power_speed")[metric].agg(["mean", "std"]).reset_index()
        ps_stats["std"] = ps_stats["std"].fillna(0)
        
        # Sadece mevcut power-speed'leri al
        ps_stats = ps_stats[ps_stats["power_speed"].isin(ps_labels)]
        
        x = np.arange(len(ps_stats))
        colors = [PS_COLORS_PROFESSIONAL.get(ps, '#4a4a4a') for ps in ps_stats["power_speed"]]
        
        bars = ax.bar(x, ps_stats["mean"], yerr=ps_stats["std"],
                     color=colors, edgecolor='black', linewidth=1.2, alpha=0.9,
                     capsize=6, error_kw={'linewidth': 1.2, 'capthick': 1.2, 'ecolor': 'black'})
        
        # X label'larını C isimlerine dönüştür
        display_labels = [PS_NAME_MAP.get(ps, ps) for ps in ps_stats["power_speed"]]
        
        # Değer ve yüzde hata etiketleri
        for i, (mean_val, std_val) in enumerate(zip(ps_stats["mean"], ps_stats["std"])):
            if mean_val != 0:
                pct_error = (std_val / mean_val) * 100
                y_pos = mean_val + std_val + 0.03 * ps_stats["mean"].max()
                
                if abs(mean_val) < 0.001 and mean_val != 0:
                    label = f"{mean_val:.1e}"
                else:
                    label = f"{mean_val:.2f}"
                
                ax.text(i, y_pos, f"{label}\n±{pct_error:.1f}%", 
                       ha='center', va='bottom', fontweight='bold', fontsize=FONT_SIZE - 2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, fontweight='bold')
        
        config = metric_config.get(metric, {"unit": "", "label": metric})
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_title(config['label'], fontweight='bold', pad=10)
        
        # Y eksenini etiketler için genişlet
        max_height = (ps_stats["mean"] + ps_stats["std"]).max()
        ax.set_ylim(bottom=0, top=max_height * 1.35)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    fig.suptitle(f"Condition Comparison - {group_name.replace('_', ' ')}", fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_folder, f"condition_comparison.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_grouped_boxplot(df_group, figures_folder, group_name, ps_labels, metric_config, alg_order):
    """Grouped box plot for key metrics comparing power-speed combinations."""
    
    key_metrics = ["porosity_%", "avg_thickness_um", "connectivity_index", "fractal_dimension"]
    available_metrics = [m for m in key_metrics if m in df_group.columns]
    
    if len(available_metrics) < 2:
        return
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        plot_data = df_group[["algorithm", "power_speed", metric]].dropna().copy()
        if plot_data.empty:
            ax.set_visible(False)
            continue
        
        plot_data["algorithm"] = (
            plot_data["algorithm"].replace(ALGORITHM_NAME_MAP)
            .astype(str).str.replace("_", " ").str.title()
        )
        
        # Her power-speed ve algoritma kombinasyonu için pozisyon belirle
        available_algs = [a for a in alg_order if a in plot_data["algorithm"].values]
        available_ps = [ps for ps in ps_labels if ps in plot_data["power_speed"].values]
        
        if not available_algs or not available_ps:
            ax.set_visible(False)
            continue
        
        positions = []
        data_to_plot = []
        colors = []
        labels = []
        
        group_width = len(available_ps) + 1
        
        for i, alg in enumerate(available_algs):
            for j, ps in enumerate(available_ps):
                pos = i * group_width + j
                positions.append(pos)
                
                subset = plot_data[(plot_data["algorithm"] == alg) & (plot_data["power_speed"] == ps)][metric].values
                data_to_plot.append(subset if len(subset) > 0 else [np.nan])
                
                colors.append(PS_COLORS_PROFESSIONAL.get(ps, '#4a4a4a'))
                ps_display = PS_NAME_MAP.get(ps, ps)
                labels.append(f"{alg}\n{ps_display}")
        
        # Box plot
        bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        
        # X ekseni etiketleri - sadece algoritma isimleri
        alg_positions = [i * group_width + (len(available_ps) - 1) / 2 for i in range(len(available_algs))]
        ax.set_xticks(alg_positions)
        ax.set_xticklabels(available_algs, fontweight='bold')
        
        config = metric_config.get(metric, {"unit": "", "label": metric})
        unit_str = f" ({config['unit']})" if config['unit'] else ""
        ax.set_ylabel(f"{config['label']}{unit_str}", fontweight='bold')
        ax.set_title(f"{config['label']}", fontweight='bold')
        
        ax.yaxis.grid(True, linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
    
    # Legend oluştur - C isimlerini kullan
    legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=PS_COLORS_PROFESSIONAL.get(ps, '#4a4a4a'), alpha=0.85) 
                     for ps in ps_labels if ps in df_group["power_speed"].values]
    legend_labels = [PS_NAME_MAP.get(ps, ps) for ps in ps_labels if ps in df_group["power_speed"].values]
    fig.legend(legend_patches, legend_labels, loc='upper right', title="Condition", framealpha=0.9)
    
    # Boş eksenleri gizle
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"Metric Comparison - {group_name.replace('_', ' ')}", fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_folder, f"boxplot_comparison.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_grouped_heatmap(df_group, figures_folder, group_name, ps_labels, metric_config, alg_order):
    """Heatmap showing mean values for each algorithm and power-speed."""
    
    metrics = ["porosity_%", "avg_thickness_um", "connectivity_index", "fractal_dimension", 
               "contrast", "energy"]
    available_metrics = [m for m in metrics if m in df_group.columns]
    
    if len(available_metrics) < 3:
        return
    
    # Pivot table oluştur
    df_plot = df_group.copy()
    df_plot["algorithm"] = (
        df_plot["algorithm"].replace(ALGORITHM_NAME_MAP)
        .astype(str).str.replace("_", " ").str.title()
    )
    
    # Algoritma + condition kombinasyonu (C isimlerini kullan)
    df_plot["condition"] = df_plot["power_speed"].map(PS_NAME_MAP).fillna(df_plot["power_speed"])
    df_plot["alg_ps"] = df_plot["algorithm"] + " (" + df_plot["condition"] + ")"
    
    # Her metrik için ortalama hesapla
    pivot_data = df_plot.groupby("alg_ps")[available_metrics].mean()
    
    # Normalize et (0-1 arası)
    pivot_normalized = (pivot_data - pivot_data.min()) / (pivot_data.max() - pivot_data.min() + 1e-10)
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Metrik isimlerini kısalt
    short_labels = [metric_config.get(m, {"label": m})["label"][:15] for m in available_metrics]
    
    sns.heatmap(
        pivot_normalized,
        annot=pivot_data.round(2),
        fmt='',
        cmap='YlOrRd',
        linewidths=0.5,
        ax=ax,
        xticklabels=short_labels,
        cbar_kws={'label': 'Normalized Value'}
    )
    
    ax.set_title(f"Algorithm × Condition Heatmap - {group_name.replace('_', ' ')}", fontweight='bold')
    ax.set_xlabel("Metric", fontweight='bold')
    ax.set_ylabel("Algorithm (Condition)", fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    
    plt.tight_layout()
    
    fig_path = os.path.join(figures_folder, f"heatmap_comparison.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_grouped_correlation(df_group, figures_folder, group_name):
    """Correlation matrix for the grouped data."""
    from scipy.stats import spearmanr
    
    metrics = [
        "porosity_%", "avg_pore_area_um2", "avg_thickness_um",
        "connectivity_index", "fractal_dimension",
        "contrast", "correlation", "energy", "homogeneity"
    ]
    
    available = [m for m in metrics if m in df_group.columns]
    data = df_group[available].dropna()
    
    if data.empty or len(available) < 3:
        return
    
    corr_matrix = data.corr(method='spearman')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    short_labels = [prettify_metric_name(m).split('(')[0].strip()[:15] for m in available]
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Spearman ρ'},
        ax=ax,
        xticklabels=short_labels,
        yticklabels=short_labels
    )
    
    ax.set_title(f"Correlation Matrix - {group_name.replace('_', ' ')}", fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    
    plt.tight_layout()
    
    fig_path = os.path.join(figures_folder, f"correlation_matrix.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_grouped_pca(df_group, figures_folder, group_name):
    """PCA visualization for the grouped data, colored by condition."""
    
    numeric_cols = df_group.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['power', 'speed', 'magnification']]
    
    if len(feature_cols) < 3:
        return
    
    X = df_group[feature_cols].dropna()
    if len(X) < 5:
        return
    
    valid_idx = X.index
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_scaled)
    
    # Profesyonel algoritma renkleri
    ALG_COLORS_PROFESSIONAL = {
        "Stucki": "#1B4F72",
        "Floyd-Steinberg": "#2E4053",
        "Jarvis": "#4A235A",
        "Standard": "#17202A",
    }
    
    # Condition ile renklendir
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sol: Condition ile renklendir
    ax = axes[0]
    for ps in df_group.loc[valid_idx, 'power_speed'].unique():
        mask = df_group.loc[valid_idx, 'power_speed'] == ps
        color = PS_COLORS_PROFESSIONAL.get(ps, '#4a4a4a')
        ps_display = PS_NAME_MAP.get(ps, ps)  # C isimlerini kullan
        ax.scatter(Xp[mask, 0], Xp[mask, 1],
                  c=color, label=ps_display,
                  s=80, alpha=0.85, edgecolors='white', linewidths=0.5)
    
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontweight="bold")
    ax.set_title("PCA - Colored by Condition", fontweight="bold")
    ax.legend(title="Condition", loc='best', framealpha=0.9)
    apply_axis_style(ax)
    
    # Sağ: Algoritma ile renklendir
    ax = axes[1]
    if 'algorithm' in df_group.columns:
        algorithms = df_group.loc[valid_idx, 'algorithm'].unique()
        for alg in algorithms:
            mask = df_group.loc[valid_idx, 'algorithm'] == alg
            display_name = ALGORITHM_NAME_MAP.get(alg, alg).replace("_", " ").title()
            color = ALG_COLORS_PROFESSIONAL.get(display_name, '#4a4a4a')
            ax.scatter(Xp[mask, 0], Xp[mask, 1],
                      c=color, label=display_name,
                      s=80, alpha=0.85, edgecolors='white', linewidths=0.5)
        ax.legend(title="Algorithm", loc='best', framealpha=0.9)
    
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontweight="bold")
    ax.set_title("PCA - Colored by Algorithm", fontweight="bold")
    apply_axis_style(ax)
    
    fig.suptitle(f"PCA Analysis - {group_name.replace('_', ' ')}", fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_folder, f"pca_analysis.png")
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', facecolor='white')
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

    # 3) 3D visualizations
    for metric in [
        "porosity_%", "avg_pore_area_um2",
        "avg_thickness_um", "connectivity_index",
        "fractal_dimension"
    ]:
        plot_3d_metric(csv_out, metric, 250, 1000000)

    # 4) Dimensionality reduction
    run_dimensionality_reduction(csv_out)

    # 5) 2D visualizations
    visualize_metrics(csv_out)
    
    # 5.5) Homogeneity visualizations
    print("\n--- Generating Homogeneity Visualizations ---")
    visualize_homogeneity(csv_out)
    
    # 5.6) Final publication figures
    generate_final_publication_figures(csv_out)
    
    # 5.7) 1000x ONLY Analysis
    print("\n--- Generating 1000x Only Analysis ---")
    analyze_1000x_only(csv_out)
    
    # 5.8) 25000x ONLY Analysis
    print("\n--- Generating 25000x Only Analysis ---")
    analyze_25000x_only(csv_out)
    
    # 5.9) PER POWER-SPEED FOLDER ANALYSIS (NEW)
    print("\n--- Generating Per Power-Speed Analysis ---")
    analyze_per_power_speed_folder(csv_out, folder)
    
    # 5.10) GROUPED POWER-SPEED ANALYSIS (20-10 & 35-20, 20-20 & 80-80)
    print("\n--- Generating Grouped Power-Speed Analysis ---")
    analyze_grouped_power_speed(csv_out, folder)
    
    # 6) Q1 STATISTICAL ANALYSIS (NEW)
    print("\n" + "="*80)
    print("Q1 PUBLICATION ANALYSIS")
    print("="*80)
    
    # Sample size report
    generate_sample_size_report(csv_out)
    
    # Outlier detection
    from outlier_detection import analyze_outliers
    outliers_dir = os.path.join(_SCRIPT_DIR, "SEM_outliers")
    analyze_outliers(csv_out, outliers_dir, method='iqr')
    
    # Statistical analysis with publication tables
    statistical_results = run_statistical_analysis(csv_out)
    
    # Generate reproducibility report
    from reproducibility_config import generate_reproducibility_report
    generate_reproducibility_report(
        os.path.join(_SCRIPT_DIR, "reproducibility_config.json")
    )
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\n✔ Results CSV: {csv_out}")
    print(f"✔ Figures: {FIG_DIR}/")
    print(f"✔ 1000x Analysis: {FIG_DIR}/1000x/")
    print(f"✔ 25000x Analysis: {FIG_DIR}/25000x/")
    print(f"✔ Per Power-Speed Analysis: Each folder (20-10, 20-20, etc.) → analysis_results/")
    print(f"✔ Grouped Analysis: SEM_grouped_analysis/ (Group1: 20-10 & 35-20, Group2: 20-20 & 80-80)")
    print(f"✔ Statistical report: SEM_statistical_report.txt")
    print(f"✔ LaTeX tables: SEM_tables/")
    print(f"✔ Outlier analysis: SEM_outliers/")
    print(f"✔ Reproducibility config: reproducibility_config.json")
    print("\n" + "="*80)
    print("FILES READY FOR Q1 PUBLICATION! 🎉")
    print("="*80)
    print("\nNext steps:")
    print("1. Review METHODS_DOCUMENTATION.md and fill in placeholders")
    print("2. Check SEM_statistical_report.txt for significant results")
    print("3. Review outlier analysis in SEM_outliers/")
    print("4. Include LaTeX tables in manuscript (SEM_tables/*.tex)")
    print("5. Add high-resolution figures from SEM_figures/")
    print("6. Review 1000x-only analysis in SEM_figures/1000x/")
    print("7. Review 25000x-only analysis in SEM_figures/25000x/")
    print("8. Review per power-speed analysis in each folder's analysis_results/")
    print("9. Review grouped analysis in SEM_grouped_analysis/")
    print("\nGood luck with your Q1 submission! 📄✨\n")

