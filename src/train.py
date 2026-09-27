"""
train.py
--------
Trains an XGBoost model to classify candidate pairs as matches or non-matches.
Uses GroupKFold split on the S1 entities to ensure the same entity doesn't appear
in both train and validation sets. Tunes the probability threshold for the F0.5 metric.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_curve

def compute_f05(precision, recall):
    # F0.5 = (1.25 * P * R) / (0.25 * P + R)
    num = 1.25 * precision * recall
    den = 0.25 * precision + recall
    # Handle div by zero
    f05 = np.zeros_like(num)
    mask = den > 0
    f05[mask] = num[mask] / den[mask]
    return f05

def train_model(features_csv, gt_tsv):
    print("Loading features...")
    df = pd.read_csv(features_csv)
    
    print("Loading ground truth for labels...")
    gt = {}
    with open(gt_tsv, 'r', encoding='utf-8') as f:
        import csv
        for row in csv.DictReader(f, delimiter='\t'):
            ids = [x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()] if row['matched_entity_ids'] else []
            gt[row['source1_entity_id']] = set(ids)
            
    # Create target label
    labels = []
    for _, row in df.iterrows():
        s1 = row['source1_entity_id']
        c2 = row['candidate_entity_id']
        is_match = 1 if c2 in gt.get(s1, set()) else 0
        labels.append(is_match)
    
    df['target'] = labels
    
    feature_cols = [
        'name_exact_match', 'addr_exact_match', 'country_match',
        'name_levenshtein', 'name_jaro_winkler', 'addr_levenshtein',
        'name_length_diff', 'addr_length_diff'
    ]
    
    # Split into train / val ensuring S1 entities are not split across sets
    print("Splitting data into train and validation sets...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['source1_entity_id']))
    
    X_train, y_train = df.iloc[train_idx][feature_cols], df.iloc[train_idx]['target']
    X_val, y_val = df.iloc[val_idx][feature_cols], df.iloc[val_idx]['target']
    
    print(f"Train size: {len(X_train):,} pairs. Val size: {len(X_val):,} pairs.")
    print(f"Positive samples in Train: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    
    print("Training scikit-learn HistGradientBoostingClassifier...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    # Calculate pos_weight for balancing since positive matches are rare
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg_count / max(1, pos_count)}
    
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_depth=5,
        class_weight=class_weight,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Predicting probabilities on validation set...")
    # predict_proba returns [prob_class_0, prob_class_1]
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # Precision-Recall curve to find optimal threshold for F0.5
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f05_scores = compute_f05(precisions[:-1], recalls[:-1]) # PR curve returns len(thresholds)+1
    
    best_idx = np.argmax(f05_scores)
    best_threshold = thresholds[best_idx]
    best_f05 = f05_scores[best_idx]
    best_p = precisions[best_idx]
    best_r = recalls[best_idx]
    
    print(f"\nOptimization Results:")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  Max Validation F0.5: {best_f05:.4f}")
    print(f"    (Precision: {best_p:.4f}, Recall: {best_r:.4f})")
    
    # Save the model and threshold
    import joblib
    import os
    os.makedirs("output", exist_ok=True)
    
    model_data = {
        'model': model,
        'threshold': best_threshold,
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, 'output/model.joblib')
    print("\nModel and threshold saved to output/model.joblib")
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    
    train_model(args.features, args.gt)
