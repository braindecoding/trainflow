#!/usr/bin/env python3
"""
Feature Importance Analysis
==========================

Analyze the importance and effectiveness of extracted features
for EEG-to-digit classification and cross-modal learning.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


def load_features():
    """Load extracted features"""
    print("📂 Loading extracted features...")
    
    with open('mindbigdata_subset_features_1500.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Combine all data for analysis
    all_features = np.vstack([
        data['training']['features'],
        data['validation']['features'],
        data['test']['features']
    ])
    
    all_labels = np.concatenate([
        data['training']['labels'],
        data['validation']['labels'],
        data['test']['labels']
    ])
    
    print(f"✅ Loaded {all_features.shape[0]} samples with {all_features.shape[1]} features")
    return all_features, all_labels, data


def analyze_feature_statistics(features, labels):
    """Analyze basic feature statistics"""
    print(f"\n📊 Feature Statistics Analysis...")
    
    # Basic statistics
    feature_means = features.mean(axis=0)
    feature_stds = features.std(axis=0)
    feature_vars = features.var(axis=0)
    
    # Feature diversity
    low_variance_features = (feature_vars < 0.01).sum()
    zero_variance_features = (feature_vars == 0).sum()
    
    print(f"   Total features: {features.shape[1]}")
    print(f"   Zero variance features: {zero_variance_features}")
    print(f"   Low variance features (< 0.01): {low_variance_features}")
    print(f"   High variance features: {features.shape[1] - low_variance_features}")
    
    # Feature range analysis
    feature_ranges = features.max(axis=0) - features.min(axis=0)
    print(f"   Feature range statistics:")
    print(f"     Min range: {feature_ranges.min():.4f}")
    print(f"     Max range: {feature_ranges.max():.4f}")
    print(f"     Mean range: {feature_ranges.mean():.4f}")
    
    return {
        'means': feature_means,
        'stds': feature_stds,
        'vars': feature_vars,
        'ranges': feature_ranges
    }


def univariate_feature_selection(features, labels, k=1000):
    """Perform univariate feature selection"""
    print(f"\n🎯 Univariate Feature Selection (top {k} features)...")
    
    # F-score based selection
    f_selector = SelectKBest(score_func=f_classif, k=k)
    f_features = f_selector.fit_transform(features, labels)
    f_scores = f_selector.scores_
    f_selected = f_selector.get_support()
    
    # Mutual information based selection
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=k)
    mi_features = mi_selector.fit_transform(features, labels)
    mi_scores = mi_selector.scores_
    mi_selected = mi_selector.get_support()
    
    print(f"   F-score selection: {f_features.shape[1]} features selected")
    print(f"   Mutual info selection: {mi_features.shape[1]} features selected")
    
    # Overlap analysis
    overlap = np.sum(f_selected & mi_selected)
    print(f"   Overlap between methods: {overlap} features ({overlap/k*100:.1f}%)")
    
    return {
        'f_scores': f_scores,
        'f_selected': f_selected,
        'mi_scores': mi_scores,
        'mi_selected': mi_selected,
        'f_features': f_features,
        'mi_features': mi_features
    }


def tree_based_feature_importance(features, labels):
    """Use Random Forest for feature importance"""
    print(f"\n🌳 Tree-based Feature Importance Analysis...")
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features, labels)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    print(f"   Random Forest accuracy: {rf.score(features, labels):.4f}")
    print(f"   Top 10 most important features:")
    for i in range(10):
        idx = indices[i]
        print(f"     Feature {idx}: {importances[idx]:.6f}")
    
    return {
        'importances': importances,
        'indices': indices,
        'model': rf
    }


def pca_analysis(features, labels):
    """Perform PCA analysis"""
    print(f"\n📐 Principal Component Analysis...")
    
    # Fit PCA
    pca = PCA()
    pca_features = pca.fit_transform(features)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Find components for different variance thresholds
    var_90 = np.argmax(cumulative_var >= 0.90) + 1
    var_95 = np.argmax(cumulative_var >= 0.95) + 1
    var_99 = np.argmax(cumulative_var >= 0.99) + 1
    
    print(f"   Components for 90% variance: {var_90}")
    print(f"   Components for 95% variance: {var_95}")
    print(f"   Components for 99% variance: {var_99}")
    print(f"   First 10 components explain: {cumulative_var[9]:.4f} variance")
    
    return {
        'pca': pca,
        'explained_var': explained_var,
        'cumulative_var': cumulative_var,
        'pca_features': pca_features,
        'components_90': var_90,
        'components_95': var_95,
        'components_99': var_99
    }


def classification_benchmark(features, labels, selection_results, pca_results):
    """Benchmark classification with different feature sets"""
    print(f"\n🏆 Classification Benchmark...")
    
    # Split data
    n_train = int(0.8 * len(features))
    indices = np.random.permutation(len(features))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    results = {}
    
    # 1. All features
    lr_all = LogisticRegression(max_iter=1000, random_state=42)
    lr_all.fit(X_train, y_train)
    acc_all = accuracy_score(y_test, lr_all.predict(X_test))
    results['all_features'] = acc_all
    print(f"   All features ({features.shape[1]}): {acc_all:.4f}")
    
    # 2. F-score selected features
    X_train_f = selection_results['f_features'][train_idx]
    X_test_f = selection_results['f_features'][test_idx]
    lr_f = LogisticRegression(max_iter=1000, random_state=42)
    lr_f.fit(X_train_f, y_train)
    acc_f = accuracy_score(y_test, lr_f.predict(X_test_f))
    results['f_selected'] = acc_f
    print(f"   F-score selected (1000): {acc_f:.4f}")
    
    # 3. Mutual info selected features
    X_train_mi = selection_results['mi_features'][train_idx]
    X_test_mi = selection_results['mi_features'][test_idx]
    lr_mi = LogisticRegression(max_iter=1000, random_state=42)
    lr_mi.fit(X_train_mi, y_train)
    acc_mi = accuracy_score(y_test, lr_mi.predict(X_test_mi))
    results['mi_selected'] = acc_mi
    print(f"   Mutual info selected (1000): {acc_mi:.4f}")
    
    # 4. PCA features (95% variance)
    n_components = pca_results['components_95']
    X_train_pca = pca_results['pca_features'][train_idx, :n_components]
    X_test_pca = pca_results['pca_features'][test_idx, :n_components]
    lr_pca = LogisticRegression(max_iter=1000, random_state=42)
    lr_pca.fit(X_train_pca, y_train)
    acc_pca = accuracy_score(y_test, lr_pca.predict(X_test_pca))
    results['pca_95'] = acc_pca
    print(f"   PCA 95% variance ({n_components}): {acc_pca:.4f}")
    
    return results


def create_visualizations(stats, selection_results, pca_results, save_dir='./feature_analysis'):
    """Create feature analysis visualizations"""
    print(f"\n📈 Creating visualizations...")
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 1. Feature variance distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(stats['vars'], bins=50, alpha=0.7, color='blue')
    ax1.set_xlabel('Feature Variance')
    ax1.set_ylabel('Count')
    ax1.set_title('Feature Variance Distribution')
    ax1.set_yscale('log')
    
    # 2. PCA explained variance
    ax2.plot(range(1, 101), pca_results['cumulative_var'][:100], 'b-', linewidth=2)
    ax2.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='90%')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax2.axhline(y=0.99, color='green', linestyle='--', alpha=0.7, label='99%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('PCA Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature importance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top F-scores
    top_f_indices = np.argsort(selection_results['f_scores'])[-20:]
    ax1.barh(range(20), selection_results['f_scores'][top_f_indices])
    ax1.set_xlabel('F-score')
    ax1.set_title('Top 20 F-scores')
    
    # Top MI scores
    top_mi_indices = np.argsort(selection_results['mi_scores'])[-20:]
    ax2.barh(range(20), selection_results['mi_scores'][top_mi_indices])
    ax2.set_xlabel('Mutual Information Score')
    ax2.set_title('Top 20 Mutual Information Scores')
    
    plt.tight_layout()
    plt.savefig(save_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Visualizations saved to: {save_path}")


def main():
    """Main feature analysis function"""
    print("🔬 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Load data
    features, labels, data = load_features()
    
    # Basic statistics
    stats = analyze_feature_statistics(features, labels)
    
    # Univariate selection
    selection_results = univariate_feature_selection(features, labels, k=1000)
    
    # Tree-based importance
    tree_results = tree_based_feature_importance(features, labels)
    
    # PCA analysis
    pca_results = pca_analysis(features, labels)
    
    # Classification benchmark
    benchmark_results = classification_benchmark(features, labels, selection_results, pca_results)
    
    # Create visualizations
    create_visualizations(stats, selection_results, pca_results)
    
    # Summary
    print(f"\n📋 FEATURE ANALYSIS SUMMARY:")
    print(f"   Original features: {features.shape[1]}")
    print(f"   Recommended for CLIP: 1000-5000 features")
    print(f"   PCA 95% variance: {pca_results['components_95']} components")
    print(f"   Best classification method: {max(benchmark_results, key=benchmark_results.get)}")
    print(f"   Best accuracy: {max(benchmark_results.values()):.4f}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   1. Use F-score or MI selection for 1000-5000 top features")
    print(f"   2. Consider PCA with {pca_results['components_95']} components")
    print(f"   3. Current 39K features may be too many for CLIP")
    print(f"   4. Feature selection could improve training efficiency")


if __name__ == "__main__":
    main()
