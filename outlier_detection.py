"""
Outlier Detection and Reporting Module
=======================================

Identifies outliers using IQR or Z-score methods and generates
comparison reports with/without outliers.
"""

import numpy as np
import pandas as pd
import os


def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Outliers are values below Q1 - multiplier*IQR or above Q3 + multiplier*IQR
    
    Args:
        data: DataFrame
        column: Column name to check for outliers
        multiplier: IQR multiplier (default 1.5 for standard outliers)
        
    Returns:
        DataFrame with outlier column added
    """
    values = data[column].dropna()
    
    if len(values) < 4:
        data[f'{column}_outlier_iqr'] = False
        return data
    
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    data[f'{column}_outlier_iqr'] = (
        (data[column] < lower_bound) | (data[column] > upper_bound)
    )
    
    return data


def detect_outliers_zscore(data, column, threshold=3):
    """
    Detect outliers using the Z-score method.
    
    Outliers are values with |Z-score| > threshold
    
    Args:
        data: DataFrame
        column: Column name to check for outliers
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        DataFrame with outlier column added
    """
    values = data[column].dropna()
    
    if len(values) < 3:
        data[f'{column}_outlier_zscore'] = False
        return data
    
    mean = values.mean()
    std = values.std()
    
    if std == 0:
        data[f'{column}_outlier_zscore'] = False
        return data
    
    z_scores = np.abs((data[column] - mean) / std)
    data[f'{column}_outlier_zscore'] = z_scores > threshold
    
    return data


def analyze_outliers(csv_path, output_dir, metrics=None, method='iqr'):
    """
    Comprehensive outlier analysis across all metrics.
    
    Args:
        csv_path: Path to results CSV
        output_dir: Directory to save reports
        metrics: List of metrics to analyze (default: all numeric)
        method: 'iqr', 'zscore', or 'both'
        
    Returns:
        dict: Summary of outliers detected
    """
    df = pd.read_csv(csv_path)
    
    if metrics is None:
        metrics = [
            "porosity_%", "avg_pore_area_um2", "avg_pore_equiv_diam_um",
            "avg_thickness_um", "connectivity_index", "fractal_dimension",
            "contrast", "correlation", "energy", "homogeneity"
        ]
    
    # Filter available metrics
    metrics = [m for m in metrics if m in df.columns]
    
    outlier_summary = []
    outlier_files = []
    
    print("\n" + "="*80)
    print("OUTLIER DETECTION REPORT")
    print("="*80 + "\n")
    
    for metric in metrics:
        print(f"\n📊 Metric: {metric}")
        print("-" * 80)
        
        # Detect outliers
        if method in ['iqr', 'both']:
            df = detect_outliers_iqr(df, metric)
            n_outliers_iqr = df[f'{metric}_outlier_iqr'].sum()
            print(f"   IQR method: {n_outliers_iqr} outliers detected")
            
            if n_outliers_iqr > 0:
                outlier_rows = df[df[f'{metric}_outlier_iqr']]
                for _, row in outlier_rows.iterrows():
                    outlier_files.append({
                        'metric': metric,
                        'method': 'IQR',
                        'file': row.get('file', 'unknown'),
                        'algorithm': row.get('algorithm', 'unknown'),
                        'value': row[metric],
                        'magnification': row.get('magnification', 'unknown')
                    })
        
        if method in ['zscore', 'both']:
            df = detect_outliers_zscore(df, metric)
            n_outliers_z = df[f'{metric}_outlier_zscore'].sum()
            print(f"   Z-score method: {n_outliers_z} outliers detected")
            
            if method == 'zscore' and n_outliers_z > 0:
                outlier_rows = df[df[f'{metric}_outlier_zscore']]
                for _, row in outlier_rows.iterrows():
                    outlier_files.append({
                        'metric': metric,
                        'method': 'Z-score',
                        'file': row.get('file', 'unknown'),
                        'algorithm': row.get('algorithm', 'unknown'),
                        'value': row[metric],
                        'magnification': row.get('magnification', 'unknown')
                    })
        
        # Summary statistics with/without outliers
        if method == 'iqr' or method == 'both':
            outlier_col = f'{metric}_outlier_iqr'
        else:
            outlier_col = f'{metric}_outlier_zscore'
        
        all_data = df[metric].dropna()
        clean_data = df[~df[outlier_col]][metric].dropna()
        
        summary = {
            'metric': metric,
            'n_total': len(all_data),
            'n_outliers': df[outlier_col].sum(),
            'outlier_pct': df[outlier_col].sum() / len(all_data) * 100 if len(all_data) > 0 else 0,
            'mean_all': all_data.mean(),
            'std_all': all_data.std(),
            'mean_clean': clean_data.mean(),
            'std_clean': clean_data.std(),
            'mean_diff_pct': abs(all_data.mean() - clean_data.mean()) / all_data.mean() * 100 
                            if all_data.mean() != 0 else 0
        }
        
        outlier_summary.append(summary)
        
        print(f"   Mean (all data): {summary['mean_all']:.3f} ± {summary['std_all']:.3f}")
        print(f"   Mean (outliers removed): {summary['mean_clean']:.3f} ± {summary['std_clean']:.3f}")
        print(f"   Difference: {summary['mean_diff_pct']:.2f}%")
    
    # Save reports
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary table
    summary_df = pd.DataFrame(outlier_summary)
    summary_df.to_csv(os.path.join(output_dir, 'outlier_summary.csv'), index=False)
    
    # Detailed outlier list
    if outlier_files:
        outlier_df = pd.DataFrame(outlier_files)
        outlier_df.to_csv(os.path.join(output_dir, 'outlier_files.csv'), index=False)
        print(f"\n✔ Outlier files list saved → {output_dir}/outlier_files.csv")
    else:
        print("\n✔ No outliers detected!")
    
    print(f"✔ Outlier summary saved → {output_dir}/outlier_summary.csv")
    
    # Generate text report
    with open(os.path.join(output_dir, 'outlier_report.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("OUTLIER DETECTION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Method: {method.upper()}\n")
        f.write(f"Total metrics analyzed: {len(metrics)}\n\n")
        
        for summary in outlier_summary:
            f.write("-" * 80 + "\n")
            f.write(f"Metric: {summary['metric']}\n")
            f.write(f"  Total observations: {summary['n_total']}\n")
            f.write(f"  Outliers detected: {summary['n_outliers']} ({summary['outlier_pct']:.1f}%)\n")
            f.write(f"  Mean (all): {summary['mean_all']:.3f} ± {summary['std_all']:.3f}\n")
            f.write(f"  Mean (clean): {summary['mean_clean']:.3f} ± {summary['std_clean']:.3f}\n")
            f.write(f"  Impact of removal: {summary['mean_diff_pct']:.2f}% change in mean\n\n")
        
        f.write("="*80 + "\n")
        f.write("RECOMMENDATION:\n")
        f.write("="*80 + "\n")
        
        high_impact = [s for s in outlier_summary if s['mean_diff_pct'] > 10]
        if high_impact:
            f.write(f"\n⚠️  HIGH IMPACT: {len(high_impact)} metrics show >10% change when outliers removed:\n")
            for s in high_impact:
                f.write(f"  - {s['metric']}: {s['mean_diff_pct']:.1f}% change\n")
            f.write("\nConsider reporting results both with and without outliers.\n")
        else:
            f.write("\n✓ MINIMAL IMPACT: Outlier removal has <10% effect on all metrics.\n")
            f.write("  Retaining outliers is acceptable as they represent real data variability.\n")
    
    print(f"✔ Outlier report saved → {output_dir}/outlier_report.txt\n")
    
    return {'summary': summary_df, 'files': outlier_df if outlier_files else None}


if __name__ == "__main__":
    # Test mode
    import numpy as np
    
    print("Outlier Detection Module - Test Mode\n")
    
    # Generate test data with outliers
    np.random.seed(42)
    test_df = pd.DataFrame({
        'file': [f'img_{i}.tif' for i in range(30)],
        'algorithm': ['Floyd'] * 10 + ['Jarvis'] * 10 + ['Stucki'] * 10,
        'porosity': np.concatenate([
            np.random.normal(15, 2, 9),
            [35],  # outlier
            np.random.normal(18, 2, 10),
            np.random.normal(16, 2, 10)
        ]),
        'magnification': [5000] * 30
    })
    
    test_df.to_csv('test_results.csv', index=False)
    
    # Run analysis
    results = analyze_outliers('test_results.csv', 'test_outliers', ['porosity'], method='iqr')
    
    print("\nTest completed. Check test_outliers/ directory for outputs.")
    
    # Cleanup
    import os
    os.remove('test_results.csv')
    
    print("\nModule loaded successfully!")
