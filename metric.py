import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

LAMBDA_1 = 0.7
LAMBDA_2 = 0.3
T_CS = 0.7          #coverage score threshold, can be changed
T_MIN_SAMPLES = 10  #min. samples threshold, can be changed

df = pd.read_csv("data.csv")
CATEGORIES = ["race", "gender", "age"]

def get_entropy(proportions):
    entropy = -sum(p * math.log(p) for p in proportions if p > 0)
    return entropy

def compute_category_scores(df, category):
    counts = df[category].value_counts()
    total = counts.sum()
    proportions = counts / total
    entropy = get_entropy(proportions)
    min_prop = min(proportions)
    return entropy, min_prop

def compute_coverage_score(df, categories, lambda1, lambda2):
    total_score = 0
    per_category_scores = {}
    for current_category in categories:
        H_i, P_i = compute_category_scores(df, current_category)
        cat_score = lambda1 * H_i + lambda2 * P_i
        per_category_scores[current_category] = cat_score
        total_score += cat_score
    final_cs = total_score / len(categories)
    return final_cs, per_category_scores

def check_minimum_representation(df, categories, T_min):
    underrepresented = {}
    for current_category in categories:
        counts = df[current_category].value_counts()
        for subcat, count in counts.items():
            if count < T_min:
                if current_category not in underrepresented:
                    underrepresented[current_category] = []
                underrepresented[current_category].append((subcat, count))
    return underrepresented

#call metric computations
coverage_score, per_category_scores = compute_coverage_score(df, CATEGORIES, LAMBDA_1, LAMBDA_2)
underrepresented = check_minimum_representation(df, CATEGORIES, T_MIN_SAMPLES)

print(f"Coverage Score (CS): {coverage_score:.4f}")
if coverage_score < T_CS:
    T_CATEGORY_CS = 0.5
    print("Dataset flagged for low overall coverage.")
    low_coverage_categories = {cat: score for cat, score in per_category_scores.items() if score < T_CATEGORY_CS}
    if low_coverage_categories:
        print("Recommend collecting more samples from categories with particularly low coverage scores:")
        for cat, score in low_coverage_categories.items():
            print(f"  - {cat}: {score:.4f} (threshold = {T_CATEGORY_CS})")

print()
print(f"Underrepresentation: {bool(underrepresented)}")
if underrepresented:
    print("Recommend collecting more samples from categories with insufficient minimum samples:")
    for category, problems in underrepresented.items():
        print(f"  - {category}:")
        for subcat, count in problems:
            print(f"    * Class {subcat} has {count} samples (threshold = {T_MIN_SAMPLES})")
