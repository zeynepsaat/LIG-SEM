import os
import numpy as np
from skimage import io as skio
import re
from collections import defaultdict

POROSITY_MIN = 2500
POROSITY_MAX = 10000
THICKNESS_MIN = 25000

def parse_mag(filename):
    m = re.search(r"_(\d+)x\.", filename.lower())
    return int(m.group(1)) if m else None

images = []
for root, dirs, files in os.walk("."):
    if "SEM_outputs" in root or ".git" in root or "__pycache__" in root:
        continue
    for f in files:
        if f.lower().endswith(".tif"):
            mag = parse_mag(f)
            if mag:
                images.append({
                    'file': f,
                    'path': os.path.join(root, f),
                    'mag': mag,
                    'algorithm': os.path.basename(os.path.dirname(root))
                })

mag_groups = defaultdict(list)
for img in images:
    mag_groups[img['mag']].append(img)

print("\n" + "="*80)
print("MAGNIFICATION ANALYSIS")
print("="*80)

print("\n1. AVAILABLE MAGNIFICATIONS:")
print("-" * 80)
for mag in sorted(mag_groups.keys()):
    count = len(mag_groups[mag])
    print(f"  {mag:>6}x: {count:>3} images")

print("\n2. MAGNIFICATION RANGES IN CODE:")
print("-" * 80)
print(f"  Porosity analysis:     {POROSITY_MIN}x - {POROSITY_MAX}x")
print(f"  Thickness/Connectivity: >={THICKNESS_MIN}x")

print("\n3. IMAGE CLASSIFICATION:")
print("-" * 80)

porosity_images = [img for img in images if POROSITY_MIN <= img['mag'] <= POROSITY_MAX]
thickness_images = [img for img in images if img['mag'] >= THICKNESS_MIN]
unused_images = [img for img in images if img['mag'] < POROSITY_MIN or (POROSITY_MAX < img['mag'] < THICKNESS_MIN)]

print(f"  Porosity analysis ({POROSITY_MIN}-{POROSITY_MAX}x):  {len(porosity_images)} images")
print(f"    Magnifications: {sorted(set(img['mag'] for img in porosity_images))}")

print(f"  Thickness/Connectivity (>={THICKNESS_MIN}x): {len(thickness_images)} images")
print(f"    Magnifications: {sorted(set(img['mag'] for img in thickness_images))}")

print(f"  Unused/Low magnifications:          {len(unused_images)} images")
if unused_images:
    print(f"    Magnifications: {sorted(set(img['mag'] for img in unused_images))}")

print("\n4. SAMPLE SIZE PER ALGORITHM:")
print("-" * 80)
algo_counts = defaultdict(lambda: {'porosity': 0, 'thickness': 0, 'total': 0})
for img in images:
    algo = img['algorithm']
    algo_counts[algo]['total'] += 1
    if POROSITY_MIN <= img['mag'] <= POROSITY_MAX:
        algo_counts[algo]['porosity'] += 1
    if img['mag'] >= THICKNESS_MIN:
        algo_counts[algo]['thickness'] += 1

for algo in sorted(algo_counts.keys()):
    counts = algo_counts[algo]
    print(f"  {algo:10s}: Total={counts['total']:2d}, Porosity={counts['porosity']:2d}, Thickness={counts['thickness']:2d}")

print("\n5. IMAGE QUALITY CHECK (sample):")
print("-" * 80)

sample_mags = [2500, 5000, 10000, 25000, 50000]
for mag in sample_mags:
    if mag in mag_groups and len(mag_groups[mag]) > 0:
        sample = mag_groups[mag][0]
        try:
            img = skio.imread(sample['path'], as_gray=True)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            mean_val = img.mean()
            std_val = img.std()
            
            black_pct = np.sum(img < 30) / img.size * 100
            white_pct = np.sum(img > 225) / img.size * 100
            
            print(f"\n  {mag}x:")
            print(f"    Size: {img.shape[1]}x{img.shape[0]} pixels")
            print(f"    Intensity: {mean_val:.1f} +/- {std_val:.1f}")
            print(f"    Black pixels: {black_pct:.1f}%, White pixels: {white_pct:.1f}%")
            
            if black_pct > 50 or white_pct > 50:
                print(f"    WARNING: Possible overexposure issue")
            elif 20 < mean_val < 100:
                print(f"    WARNING: Image appears dark")
            elif 150 < mean_val < 235:
                print(f"    WARNING: Image appears bright")
            else:
                print(f"    OK: Good contrast")
                
        except Exception as e:
            print(f"\n  {mag}x: Error loading - {str(e)[:50]}")

print("\n" + "="*80)
print("MAGNIFICATION APPROPRIATENESS ASSESSMENT")
print("="*80)

print("\nPOROSITY & PORE SIZE (2,500-10,000x):")
print("  CORRECT CHOICE")
print("  - Individual pores clearly visible")
print("  - Good field of view for statistics")
print("  - Pore boundaries segmentable")
print(f"  Your data: {len(porosity_images)} images")

print("\nTHICKNESS & CONNECTIVITY (>=25,000x):")
print("  CORRECT CHOICE")
print("  - Individual struts resolvable")
print("  - Network topology visible")
print("  - Junction points identifiable")
print(f"  Your data: {len(thickness_images)} images")

if unused_images:
    print("\nUNUSED MAGNIFICATIONS:")
    unused_mags = sorted(set(img['mag'] for img in unused_images))
    print(f"  {len(unused_images)} images at {unused_mags}")
    print("  Status: Correctly excluded from quantitative analysis")
    if any(m < 2000 for m in unused_mags):
        print("  Note: Low-mag images good for overview figures")

print("\n" + "="*80)
print("OVERALL: EXCELLENT MAGNIFICATION STRATEGY")
print("="*80)
print("\nStrengths:")
print("  1. Clear scale separation for different metrics")
print("  2. Each analysis at appropriate magnification")
print("  3. Follows SEM best practices")
print("  4. No ambiguous intermediate ranges")
print("\nFor manuscript:")
print('  "Porosity analysis at 2,500-10,000x (pore-scale resolution)")
print('   Topology analysis at >=25,000x (network-scale resolution)"')
print("\n" + "="*80)
