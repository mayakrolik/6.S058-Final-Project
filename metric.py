import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns 

# Parameters NEED TO CHANGE
T_CS = 0.7           # Cross-Category Coverage Score threshold
T_LOW_COVERAGE = 0.2 # Threshold for low normalized coefficient
T_MIN_SAMPLES = 5    # Minimum raw count per group

df = pd.read_csv("test_outputs_large.csv")

AGE_LUMPING = {
    '3-9': 'Child and Young Adult',
    '10-19': 'Child and Young Adult',
    '20-29': 'Child and Young Adult',
    '30-39': 'Adult',
    '40-49': 'Adult',
    '50-59': 'Senior',
    '60-69': 'Senior',
    '70+': 'Senior'
}

RACE_LUMPING = {
    'White': 'White',
    'Black': 'Black',
    'Latino_Hispanic': 'Latino_Hispanic',
    'East Asian': 'Asian',
    'Southeast Asian': 'Asian',
    'Indian': 'Asian',
    'Middle Eastern': 'White'
}

# Map and transform
if 'age' not in df.columns:
    raise ValueError("Input CSV missing required 'age' column.")
if 'race' not in df.columns:
    raise ValueError("Input CSV missing required 'race' column.")

df['age_group_lumped'] = df['age'].map(AGE_LUMPING)
if df['age_group_lumped'].isnull().any():
    missing_values = df.loc[df['age_group_lumped'].isnull(), 'age'].unique()
    raise ValueError(f"Unmapped age values found: {missing_values}")

df['race_lumped'] = df['race'].map(RACE_LUMPING)
if df['race_lumped'].isnull().any():
    missing_races = df.loc[df['race_lumped'].isnull(), 'race'].unique()
    raise ValueError(f"Unmapped race values found: {missing_races}")

#Define all categories
ALL_CATEGORIES = {
    'race_lumped': ['White', 'Black', 'Latino_Hispanic', 'Asian'],
    'gender': ['Male', 'Female'],
    'age_group_lumped': ['Child and Young Adult', 'Adult', 'Senior']
}

def compute_cross_category_scores(df, all_categories, t_low_coverage):
    category_keys = list(all_categories.keys())
    all_combinations = list(itertools.product(*all_categories.values()))

    df['group_key'] = df[category_keys].astype(str).agg('-'.join, axis=1)
    group_counts = df['group_key'].value_counts().to_dict()

    raw_counts = {}
    for comb in all_combinations:
        key = '-'.join(str(x) for x in comb)
        raw_counts[key] = group_counts.get(key, 0)

    max_count = max(raw_counts.values()) if raw_counts else 1
    normalized_coeffs = {k: v / max_count for k, v in raw_counts.items()}

    normalized_values = list(normalized_coeffs.values())
    min_coeff = min(normalized_values)
    avg_coeff = np.mean(normalized_values)
    final_cs = 0.5 * min_coeff + 0.5 * avg_coeff
    # final_cs = avg_coeff  # Only average
    low_coverage_groups = {k: v for k, v in normalized_coeffs.items() if v < t_low_coverage}

    return final_cs, normalized_coeffs, low_coverage_groups

def check_minimum_representation(df, all_categories, t_min_samples):
    category_keys = list(all_categories.keys())
    df['group_key'] = df[category_keys].astype(str).agg('-'.join, axis=1)
    counts = df['group_key'].value_counts()

    underrepresented = {}
    for group, count in counts.items():
        if count < t_min_samples:
            underrepresented[group] = count
    return underrepresented

def plot_heatmap(normalized_coeffs, all_categories):
    """
    Plot heatmaps for various category pairs based on normalized coefficients.
    """

    def prepare_heatmap(x_cat, y_cat):
        x_labels = all_categories[x_cat]
        y_labels = all_categories[y_cat]

        # Initialize matrix
        matrix = np.zeros((len(y_labels), len(x_labels)))

        for i, y in enumerate(y_labels):
            for j, x in enumerate(x_labels):
                key_parts = {x_cat: x, y_cat: y}
                # Find all matching keys from the full combination keys
                matching_keys = [k for k in normalized_coeffs if x in k and y in k]
                if matching_keys:
                    matrix[i, j] = normalized_coeffs[matching_keys[0]]
                else:
                    matrix[i, j] = np.nan  # Use NaN for missing combos

        return matrix, x_labels, y_labels

    pairs = [
        ('race_lumped', 'gender'),
        ('race_lumped', 'age_group_lumped'),
        ('gender', 'age_group_lumped')
    ]

    for x_cat, y_cat in pairs:
        heatmap_data, x_labels, y_labels = prepare_heatmap(x_cat, y_cat)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_data,
            xticklabels=x_labels,
            yticklabels=y_labels,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0, vmax=1,
            mask=np.isnan(heatmap_data)
        )
        plt.title(f"Coverage Heatmap: {x_cat} vs {y_cat}")
        plt.xlabel(x_cat)
        plt.ylabel(y_cat)
        plt.tight_layout()
        plt.show()


# Run Metric Computations
coverage_score, per_group_scores, low_coverage_groups = compute_cross_category_scores(df, ALL_CATEGORIES, T_LOW_COVERAGE)
underrepresented = check_minimum_representation(df, ALL_CATEGORIES, T_MIN_SAMPLES)

print(f"Coverage Score (CS): {coverage_score:.4f}")
if coverage_score < T_CS:
    print("Dataset flagged for low overall cross-category coverage.")
    sorted_low = sorted(low_coverage_groups.items(), key=lambda x: x[1])
    print("Low-coverage groups (particularly underrepresented):")
    for group, score in sorted_low[:10]:
        print(f"  - {group}: {score:.4f}")

print()
print(f"Underrepresentation based on minimum samples: {bool(underrepresented)}")
if underrepresented:
    print("Dataset flagged for insufficient minimum samples in some groups:")
    for group, count in underrepresented.items():
        print(f"  - {group}: {count} samples (threshold = {T_MIN_SAMPLES})")

plot_heatmap(per_group_scores, ALL_CATEGORIES)

