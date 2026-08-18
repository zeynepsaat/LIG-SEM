"""
Statistical Analysis Module for Q1 Publication
===============================================

Provides comprehensive statistical tests for comparing LIG properties
across different error diffusion algorithms.

Author: Generated for LIG-SEM Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene, f_oneway, kruskal
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 1) Normality and Homogeneity Tests
# ============================================================

def test_normality(data_dict, alpha=0.05):
    """
    Test normality assumption for each group using Shapiro-Wilk test.
    
    Args:
        data_dict: {group_name: array_of_values}
        alpha: Significance level (default 0.05)
        
    Returns:
        dict: {group_name: {'statistic': W, 'p_value': p, 'is_normal': bool}}
    """
    results = {}
    for group_name, values in data_dict.items():
        values = np.array(values)
        values = values[~np.isnan(values)]
        
        if len(values) < 3:
            results[group_name] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'is_normal': None,
                'note': 'Sample size < 3'
            }
            continue
            
        stat, p = shapiro(values)
        results[group_name] = {
            'statistic': stat,
            'p_value': p,
            'is_normal': p > alpha
        }
    
    return results


def test_homogeneity(data_dict, alpha=0.05):
    """
    Test homogeneity of variances using Levene's test.
    
    Args:
        data_dict: {group_name: array_of_values}
        alpha: Significance level
        
    Returns:
        dict: {'statistic': F, 'p_value': p, 'is_homogeneous': bool}
    """
    groups = []
    for values in data_dict.values():
        values = np.array(values)
        values = values[~np.isnan(values)]
        if len(values) > 0:
            groups.append(values)
    
    if len(groups) < 2:
        return {'statistic': np.nan, 'p_value': np.nan, 
                'is_homogeneous': None, 'note': 'Insufficient groups'}
    
    stat, p = levene(*groups)
    return {
        'statistic': stat,
        'p_value': p,
        'is_homogeneous': p > alpha
    }


# ============================================================
# 2) ANOVA / Kruskal-Wallis
# ============================================================

def compare_groups(data_dict, alpha=0.05, test='auto'):
    """
    Compare multiple groups using ANOVA or Kruskal-Wallis test.
    
    Args:
        data_dict: {group_name: array_of_values}
        alpha: Significance level
        test: 'anova', 'kruskal', or 'auto' (automatic selection)
        
    Returns:
        dict: Test results with statistic, p-value, effect size
    """
    groups = []
    group_names = []
    
    for name, values in data_dict.items():
        values = np.array(values)
        values = values[~np.isnan(values)]
        if len(values) > 0:
            groups.append(values)
            group_names.append(name)
    
    if len(groups) < 2:
        return {
            'test': 'none',
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'note': 'Insufficient groups for comparison'
        }
    
    # Determine which test to use
    if test == 'auto':
        # Check normality and homogeneity
        norm_results = test_normality(data_dict, alpha)
        all_normal = all(r.get('is_normal', False) for r in norm_results.values() 
                        if r.get('is_normal') is not None)
        
        homo_result = test_homogeneity(data_dict, alpha)
        is_homogeneous = homo_result.get('is_homogeneous', False)
        
        # Use ANOVA if assumptions met, otherwise Kruskal-Wallis
        use_parametric = all_normal and is_homogeneous
        test = 'anova' if use_parametric else 'kruskal'
    
    # Perform test
    if test == 'anova':
        stat, p = f_oneway(*groups)
        effect_size = calculate_eta_squared(groups)
        test_name = 'One-way ANOVA'
    else:
        stat, p = kruskal(*groups)
        effect_size = calculate_epsilon_squared(groups)
        test_name = 'Kruskal-Wallis H-test'
    
    return {
        'test': test_name,
        'test_type': test,
        'statistic': stat,
        'p_value': p,
        'significant': p < alpha,
        'effect_size': effect_size,
        'group_names': group_names
    }


# ============================================================
# 3) Effect Size Calculations
# ============================================================

def calculate_eta_squared(groups):
    """
    Calculate eta-squared (η²) effect size for ANOVA.
    
    η² = SS_between / SS_total
    """
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    
    # Total sum of squares
    ss_total = np.sum((all_data - grand_mean)**2)
    
    if ss_total == 0:
        return 0.0
    
    return ss_between / ss_total


def calculate_epsilon_squared(groups):
    """
    Calculate epsilon-squared (ε²) effect size for Kruskal-Wallis.
    
    ε² = H / (n² - 1) / (n + 1)
    where H is Kruskal-Wallis statistic
    """
    all_data = np.concatenate(groups)
    n = len(all_data)
    k = len(groups)
    
    stat, _ = kruskal(*groups)
    
    epsilon_sq = stat / ((n**2 - 1) / (n + 1))
    
    return epsilon_sq


def cohens_d(group1, group2):
    """
    Calculate Cohen's d for pairwise comparison.
    
    d = (M1 - M2) / pooled_std
    """
    g1 = np.array(group1)
    g2 = np.array(group2)
    
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def interpret_effect_size(value, metric='eta_squared'):
    """
    Interpret effect size magnitude.
    
    Cohen's guidelines:
    - η² or ε²: small=0.01, medium=0.06, large=0.14
    - Cohen's d: small=0.2, medium=0.5, large=0.8
    """
    if np.isnan(value):
        return 'undefined'
    
    value = abs(value)
    
    if metric in ['eta_squared', 'epsilon_squared']:
        if value < 0.01:
            return 'negligible'
        elif value < 0.06:
            return 'small'
        elif value < 0.14:
            return 'medium'
        else:
            return 'large'
    elif metric == 'cohens_d':
        if value < 0.2:
            return 'negligible'
        elif value < 0.5:
            return 'small'
        elif value < 0.8:
            return 'medium'
        else:
            return 'large'
    else:
        return 'unknown'


# ============================================================
# 4) Post-hoc Tests
# ============================================================

def posthoc_pairwise(data_dict, test_type='auto', alpha=0.05):
    """
    Perform pairwise post-hoc comparisons.
    
    Args:
        data_dict: {group_name: array_of_values}
        test_type: 'tukey' (parametric), 'dunn' (non-parametric), or 'auto'
        alpha: Significance level
        
    Returns:
        DataFrame with pairwise comparison results
    """
    from scipy.stats import mannwhitneyu, ttest_ind
    from statsmodels.stats.multitest import multipletests
    
    group_names = list(data_dict.keys())
    pairs = list(combinations(group_names, 2))
    
    results = []
    
    for g1_name, g2_name in pairs:
        g1 = np.array(data_dict[g1_name])
        g2 = np.array(data_dict[g2_name])
        
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        
        if len(g1) < 2 or len(g2) < 2:
            results.append({
                'group1': g1_name,
                'group2': g2_name,
                'statistic': np.nan,
                'p_value': np.nan,
                'cohens_d': np.nan,
                'significant': False,
                'note': 'Insufficient data'
            })
            continue
        
        # Choose test
        if test_type == 'auto':
            # Use Mann-Whitney U (more conservative, robust)
            stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
            test_used = 'Mann-Whitney U'
        elif test_type == 'parametric':
            stat, p = ttest_ind(g1, g2)
            test_used = 't-test'
        else:
            stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
            test_used = 'Mann-Whitney U'
        
        d = cohens_d(g1, g2)
        
        results.append({
            'group1': g1_name,
            'group2': g2_name,
            'test': test_used,
            'statistic': stat,
            'p_value': p,
            'cohens_d': d,
            'effect_size_interpretation': interpret_effect_size(d, 'cohens_d'),
            'mean_diff': np.mean(g1) - np.mean(g2)
        })
    
    df = pd.DataFrame(results)
    
    # Apply multiple testing correction (Bonferroni)
    if len(df) > 0 and 'p_value' in df.columns:
        valid_p = ~df['p_value'].isna()
        if valid_p.sum() > 0:
            _, p_corrected, _, _ = multipletests(
                df.loc[valid_p, 'p_value'], 
                alpha=alpha, 
                method='bonferroni'
            )
            df.loc[valid_p, 'p_corrected'] = p_corrected
            df.loc[valid_p, 'significant'] = p_corrected < alpha
        else:
            df['p_corrected'] = np.nan
            df['significant'] = False
    
    return df


# ============================================================
# 5) Full Analysis Pipeline
# ============================================================

def analyze_metric_by_algorithm(df, metric, algorithm_col='algorithm', alpha=0.05):
    """
    Complete statistical analysis for one metric across algorithms.
    
    Args:
        df: DataFrame with data
        metric: Column name of the metric to analyze
        algorithm_col: Column name for grouping variable
        alpha: Significance level
        
    Returns:
        dict: Complete analysis results
    """
    # Prepare data
    data_dict = {}
    for alg in df[algorithm_col].unique():
        values = df[df[algorithm_col] == alg][metric].dropna().values
        if len(values) > 0:
            data_dict[alg] = values
    
    if len(data_dict) < 2:
        return {
            'metric': metric,
            'note': 'Insufficient groups for analysis',
            'n_groups': len(data_dict)
        }
    
    # Descriptive statistics
    descriptive = {}
    for name, values in data_dict.items():
        descriptive[name] = {
            'n': len(values),
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
    
    # Assumption tests
    normality = test_normality(data_dict, alpha)
    homogeneity = test_homogeneity(data_dict, alpha)
    
    # Main comparison test
    comparison = compare_groups(data_dict, alpha, test='auto')
    
    # Post-hoc if significant
    posthoc = None
    if comparison.get('significant', False):
        posthoc = posthoc_pairwise(data_dict, test_type='auto', alpha=alpha)
    
    return {
        'metric': metric,
        'descriptive': descriptive,
        'normality': normality,
        'homogeneity': homogeneity,
        'comparison': comparison,
        'posthoc': posthoc,
        'alpha': alpha
    }


def format_p_value(p):
    """Format p-value for publication (e.g., p < 0.001)."""
    if np.isnan(p):
        return 'n/a'
    elif p < 0.001:
        return 'p < 0.001***'
    elif p < 0.01:
        return f'p = {p:.3f}**'
    elif p < 0.05:
        return f'p = {p:.3f}*'
    else:
        return f'p = {p:.3f}'


def format_significance_stars(p):
    """Return significance stars for plotting."""
    if np.isnan(p):
        return ''
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


# ============================================================
# 6) Publication Table Generators
# ============================================================

def generate_descriptive_table(results_dict, latex=True):
    """
    Generate descriptive statistics table for publication.
    
    Args:
        results_dict: {metric_name: analysis_results}
        latex: If True, output LaTeX format
        
    Returns:
        str: Formatted table
    """
    rows = []
    
    for metric_name, results in results_dict.items():
        if 'descriptive' not in results:
            continue
            
        for alg_name, stats in results['descriptive'].items():
            rows.append({
                'Metric': metric_name,
                'Algorithm': alg_name,
                'n': stats['n'],
                'Mean': f"{stats['mean']:.2f}",
                'SD': f"{stats['std']:.2f}",
                'Median': f"{stats['median']:.2f}",
                'Min': f"{stats['min']:.2f}",
                'Max': f"{stats['max']:.2f}"
            })
    
    df = pd.DataFrame(rows)
    
    if latex:
        return df.to_latex(index=False, escape=False)
    else:
        return df.to_string(index=False)


def generate_anova_table(results_dict, latex=True):
    """
    Generate ANOVA/Kruskal-Wallis results table.
    
    Args:
        results_dict: {metric_name: analysis_results}
        latex: If True, output LaTeX format
        
    Returns:
        str: Formatted table
    """
    rows = []
    
    for metric_name, results in results_dict.items():
        if 'comparison' not in results:
            continue
        
        comp = results['comparison']
        
        rows.append({
            'Metric': metric_name,
            'Test': comp.get('test', 'N/A'),
            'Statistic': f"{comp.get('statistic', np.nan):.3f}",
            'p-value': format_p_value(comp.get('p_value', np.nan)),
            'Effect Size': f"{comp.get('effect_size', np.nan):.3f}",
            'Significant': 'Yes' if comp.get('significant', False) else 'No'
        })
    
    df = pd.DataFrame(rows)
    
    if latex:
        return df.to_latex(index=False, escape=False)
    else:
        return df.to_string(index=False)


# ============================================================
# 7) Export Functions
# ============================================================

def save_statistical_report(results_dict, output_path):
    """
    Save comprehensive statistical report to text file.
    
    Args:
        results_dict: {metric_name: analysis_results}
        output_path: Path to save report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("LIG Error Diffusion Algorithm Comparison\n")
        f.write("=" * 80 + "\n\n")
        
        for metric_name, results in results_dict.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"METRIC: {metric_name}\n")
            f.write(f"{'=' * 80}\n\n")
            
            # Descriptive statistics
            if 'descriptive' in results:
                f.write("Descriptive Statistics:\n")
                f.write("-" * 80 + "\n")
                for alg, stats in results['descriptive'].items():
                    f.write(f"  {alg}:\n")
                    f.write(f"    n = {stats['n']}\n")
                    f.write(f"    Mean ± SD = {stats['mean']:.3f} ± {stats['std']:.3f}\n")
                    f.write(f"    Median (IQR) = {stats['median']:.3f} ")
                    f.write(f"({stats['q25']:.3f} - {stats['q75']:.3f})\n")
                    f.write(f"    Range = [{stats['min']:.3f}, {stats['max']:.3f}]\n\n")
            
            # Comparison test
            if 'comparison' in results:
                comp = results['comparison']
                f.write(f"Main Comparison Test: {comp.get('test', 'N/A')}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Statistic = {comp.get('statistic', np.nan):.4f}\n")
                f.write(f"  p-value = {comp.get('p_value', np.nan):.4f}\n")
                f.write(f"  Effect size = {comp.get('effect_size', np.nan):.4f}\n")
                f.write(f"  Significant: {comp.get('significant', False)}\n\n")
            
            # Post-hoc
            if results.get('posthoc') is not None:
                f.write("Post-hoc Pairwise Comparisons:\n")
                f.write("-" * 80 + "\n")
                posthoc = results['posthoc']
                for _, row in posthoc.iterrows():
                    f.write(f"  {row['group1']} vs {row['group2']}:\n")
                    f.write(f"    p-corrected = {row.get('p_corrected', np.nan):.4f} ")
                    f.write(f"({format_significance_stars(row.get('p_corrected', np.nan))})\n")
                    f.write(f"    Cohen's d = {row.get('cohens_d', np.nan):.3f} ")
                    f.write(f"({row.get('effect_size_interpretation', 'N/A')})\n")
                    f.write(f"    Mean difference = {row.get('mean_diff', np.nan):.3f}\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✔ Statistical report saved → {output_path}")


if __name__ == "__main__":
    # Test with sample data
    print("Statistical Analysis Module - Test Mode")
    
    # Generate sample data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'algorithm': ['Floyd'] * 10 + ['Jarvis'] * 10 + ['Stucki'] * 10,
        'porosity': np.concatenate([
            np.random.normal(15, 2, 10),
            np.random.normal(18, 2, 10),
            np.random.normal(16, 2, 10)
        ])
    })
    
    # Run analysis
    results = analyze_metric_by_algorithm(test_df, 'porosity')
    
    print("\nTest Results:")
    print(f"Test: {results['comparison']['test']}")
    print(f"p-value: {results['comparison']['p_value']:.4f}")
    print(f"Effect size: {results['comparison']['effect_size']:.4f}")
    print("\nModule loaded successfully!")
