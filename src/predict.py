"""
predict.py
----------
Loads the trained model and features, predicts matches using the optimized threshold,
and generates the final matching_results.tsv in the exact format required for submission.

Usage:
    python3 predict.py \\
        --s1 dataset/test/test_source1.tsv \\
        --features output/test_features.csv \\
        --model output/model.joblib \\
        --out output/matching_results.tsv
"""

import argparse
import csv
import pandas as pd
import joblib
from collections import defaultdict
import sys

csv.field_size_limit(sys.maxsize)

def generate_predictions(s1_path, features_path, model_path, out_path):
    print("Loading saved model and threshold...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    threshold = model_data['threshold']
    feature_cols = model_data['feature_cols']
    
    print(f"Loaded model. Using strict F0.5 threshold: {threshold:.4f}")
    
    print("Loading test features...")
    df = pd.read_csv(features_path)
    
    print("Running inference...")
    # Get probability of class 1 (match)
    probs = model.predict_proba(df[feature_cols])[:, 1]
    
    # Filter by threshold
    df['is_match'] = (probs >= threshold).astype(int)
    
    # Group matches by S1 entity
    matches = defaultdict(list)
    match_df = df[df['is_match'] == 1]
    
    for _, row in match_df.iterrows():
        matches[row['source1_entity_id']].append(row['candidate_entity_id'])
        
    print("Writing final submission file...")
    # The hackathon requires EVERY S1 entity from the test set to be in the output,
    # even if it has no matches (singletons)
    
    total_s1 = 0
    matched_s1 = 0
    total_links = 0
    
    with open(s1_path, 'r', encoding='utf-8') as f_in, \
         open(out_path, 'w', newline='', encoding='utf-8') as f_out:
         
        reader = csv.DictReader(f_in, delimiter='\t')
        writer = csv.writer(f_out, delimiter='\t')
        
        # Header required by validate_submission.py
        writer.writerow(['source1_entity_id', 'matched_entity_ids'])
        
        for row in reader:
            s1_id = row['entity_id']
            total_s1 += 1
            
            matched_ids = matches.get(s1_id, [])
            if matched_ids:
                matched_s1 += 1
                total_links += len(matched_ids)
                
            writer.writerow([s1_id, ",".join(sorted(matched_ids))])
            
    print(f"Done! Created {out_path}")
    print("-" * 40)
    print(f"Total S1 entities processed: {total_s1:,}")
    print(f"S1 entities with at least 1 match: {matched_s1:,} ({(matched_s1/total_s1)*100:.1f}%)")
    print(f"Total match links predicted: {total_links:,}")
    print("-" * 40)
    print("Ready for submission!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    generate_predictions(args.s1, args.features, args.model, args.out)
