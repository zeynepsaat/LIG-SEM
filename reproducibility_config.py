"""
Reproducibility Configuration
==============================

This file documents all parameters, settings, and versions used in the analysis
to ensure full reproducibility of results.

Generated automatically by the analysis pipeline.
"""

import sys
import numpy as np
import pandas as pd
import cv2
import skimage
import scipy
import matplotlib
import seaborn as sns
from datetime import datetime
import json
import os


def generate_reproducibility_report(output_path):
    """
    Generate comprehensive reproducibility report.
    
    Includes:
    - Software versions
    - Analysis parameters
    - Random seeds
    - Hardware info
    - Timestamp
    """
    
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "analysis_type": "LIG Error Diffusion Algorithm Comparison",
            "report_version": "1.0"
        },
        
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
            "scikit-image": skimage.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__
        },
        
        "analysis_parameters": {
            "image_preprocessing": {
                "crop_ratio": 0.88,
                "description": "Bottom 12% cropped to remove metadata panel"
            },
            
            "porosity_segmentation": {
                "gaussian_blur_sigma": 1.0,
                "adaptive_threshold_method": "gaussian",
                "block_size": 25,
                "offset": 2,
                "magnification_range": "2500-10000x",
                "description": "Adaptive local thresholding for pore detection"
            },
            
            "thickness_analysis": {
                "gaussian_blur_sigma": 0.6,
                "threshold_method": "otsu",
                "skeletonization_method": "Zhang-Suen",
                "distance_transform": "Euclidean",
                "magnification_threshold": "≥25000x",
                "description": "Morphological skeleton + distance transform"
            },
            
            "connectivity_analysis": {
                "junction_detection_window": "3x3",
                "minimum_neighbors": 3,
                "description": "Junction points identified as skeleton pixels with ≥3 neighbors"
            },
            
            "fractal_dimension": {
                "method": "box-counting",
                "box_sizes": "2^k where k = log2(min_dim) down to 2",
                "description": "Box-counting method with logarithmic regression"
            },
            
            "glcm_texture": {
                "distances": [5],
                "angles": [0, 45, 90, 135],
                "angles_radians": [0, "π/4", "π/2", "3π/4"],
                "levels": 256,
                "symmetric": True,
                "normed": True,
                "features": ["contrast", "correlation", "energy", "homogeneity"],
                "description": "4-direction rotation-invariant GLCM"
            },
            
            "scale_calibration": {
                "method": "OCR + fallback",
                "hfw_constant_um": 414000.0,
                "description": "HFW extracted via OCR; if failed, estimated as 414000/magnification"
            }
        },
        
        "statistical_parameters": {
            "significance_level": 0.05,
            "normality_test": "Shapiro-Wilk",
            "homogeneity_test": "Levene",
            "parametric_test": "One-way ANOVA",
            "nonparametric_test": "Kruskal-Wallis H-test",
            "posthoc_test": "Mann-Whitney U with Bonferroni correction",
            "effect_sizes": {
                "overall_parametric": "eta-squared (η²)",
                "overall_nonparametric": "epsilon-squared (ε²)",
                "pairwise": "Cohen's d"
            },
            "correlation_method": "Spearman rank correlation",
            "multiple_testing_correction": "Bonferroni (posthoc), FDR (correlations)"
        },
        
        "dimensionality_reduction": {
            "pca": {
                "n_components": 2,
                "scaling": "StandardScaler (z-score)",
                "description": "Principal Component Analysis"
            },
            "tsne": {
                "n_components": 2,
                "perplexity": 20,
                "random_state": 42,
                "scaling": "StandardScaler (z-score)",
                "description": "t-SNE with fixed random seed for reproducibility"
            }
        },
        
        "visualization_parameters": {
            "dpi": 1000,
            "font_size": 16,
            "figure_format": "PNG",
            "color_palette": "Reds (sequential)",
            "description": "High-resolution publication-quality figures"
        },
        
        "random_seeds": {
            "tsne": 42,
            "note": "t-SNE is the only stochastic method; seed fixed for reproducibility"
        },
        
        "computational_environment": {
            "platform": sys.platform,
            "python_implementation": sys.implementation.name,
            "note": "Hardware details not captured automatically"
        },
        
        "file_structure": {
            "input": {
                "directory_structure": "[power-speed]/[algorithm]/[images]",
                "supported_formats": [".png", ".jpg", ".jpeg", ".tif", ".bmp"],
                "magnification_source": "filename suffix (_5000x.tif) or folder name"
            },
            "outputs": {
                "main_results": "SEM_results_full.csv",
                "per_condition_results": "SEM_results_by_power_speed/",
                "figures": "SEM_figures/",
                "visual_overlays": "SEM_outputs/",
                "statistical_report": "SEM_statistical_report.txt",
                "latex_tables": "SEM_tables/",
                "methods_documentation": "METHODS_DOCUMENTATION.md",
                "reproducibility_report": "reproducibility_config.json"
            }
        },
        
        "quality_control": {
            "minimum_sample_size_warning": 3,
            "outlier_detection_methods": ["IQR (multiplier=1.5)", "Z-score (threshold=3.0)"],
            "scale_calibration_validation": "Manual verification of OCR results recommended",
            "image_quality_checks": "None automated - visual inspection recommended"
        },
        
        "known_limitations": [
            "OCR-based HFW extraction may fail for poor quality metadata",
            "Adaptive threshold parameters chosen empirically (not optimized per image)",
            "No ground truth segmentation validation performed",
            "Small sample sizes may limit statistical power",
            "Assumes all images from same SEM with consistent calibration"
        ],
        
        "citation_recommendations": {
            "software": [
                "Python (Van Rossum & Drake, 2009)",
                "NumPy (Harris et al., 2020)",
                "scikit-image (van der Walt et al., 2014)",
                "SciPy (Virtanen et al., 2020)",
                "Matplotlib (Hunter, 2007)",
                "Seaborn (Waskom, 2021)",
                "OpenCV (Bradski, 2000)"
            ],
            "methods": [
                "Otsu threshold (Otsu, 1979)",
                "Zhang-Suen skeletonization (Zhang & Suen, 1984)",
                "GLCM texture analysis (Haralick et al., 1973)",
                "Box-counting fractal dimension (Mandelbrot, 1982)",
                "Shapiro-Wilk test (Shapiro & Wilk, 1965)",
                "Kruskal-Wallis test (Kruskal & Wallis, 1952)",
                "Mann-Whitney U test (Mann & Whitney, 1947)",
                "Effect sizes (Cohen, 1988)"
            ]
        }
    }
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✔ Reproducibility report saved → {output_path}")
    
    # Also save as human-readable text
    text_path = output_path.replace('.json', '.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPRODUCIBILITY CONFIGURATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {report['metadata']['generated_at']}\n")
        f.write(f"Analysis: {report['metadata']['analysis_type']}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("SOFTWARE VERSIONS\n")
        f.write("-"*80 + "\n")
        for pkg, ver in report['software_versions'].items():
            f.write(f"  {pkg}: {ver}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("ANALYSIS PARAMETERS\n")
        f.write("-"*80 + "\n")
        for category, params in report['analysis_parameters'].items():
            f.write(f"\n{category.replace('_', ' ').title()}:\n")
            for key, val in params.items():
                if key != 'description':
                    f.write(f"  {key}: {val}\n")
            if 'description' in params:
                f.write(f"  → {params['description']}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("STATISTICAL PARAMETERS\n")
        f.write("-"*80 + "\n")
        for key, val in report['statistical_parameters'].items():
            if isinstance(val, dict):
                f.write(f"{key}:\n")
                for k, v in val.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {val}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("KNOWN LIMITATIONS\n")
        f.write("-"*80 + "\n")
        for i, limitation in enumerate(report['known_limitations'], 1):
            f.write(f"{i}. {limitation}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✔ Human-readable report saved → {text_path}")
    
    return report


if __name__ == "__main__":
    print("Reproducibility Report Generator\n")
    report = generate_reproducibility_report("reproducibility_config.json")
    print("\nReport generated successfully!")
    print("\nKey software versions:")
    for pkg, ver in report['software_versions'].items():
        print(f"  {pkg}: {ver}")
