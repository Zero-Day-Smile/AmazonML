#!/bin/bash
set -e

echo "[1/4] Blocking 50k Training Queries (Hard Negatives)..."
python3 fast_block.py \
    --s1 dataset/train/train_source1_50k.tsv \
    --s2 dataset/train/train_source2.tsv \
    --s3 dataset/train/train_source3.tsv \
    --out output/train_cands_50k.tsv \
    --topk 30

echo "[2/4] Generating Features for 50k..."
python3 fast_features.py \
    --s1 dataset/train/train_source1_50k.tsv \
    --targets "dataset/train/train_source2.tsv,dataset/train/train_source3.tsv" \
    --candidates output/train_cands_50k.tsv \
    --out output/train_features_50k.csv \
    --header "1"

echo "[3/4] Training Model on 50k queries..."
python3 train.py \
    --features output/train_features_50k.csv \
    --gt dataset/train/train_ground_truth.tsv

echo "[4/4] Predicting Test Set..."
python3 predict.py \
    --s1 dataset/test/test_source1.tsv \
    --features output/test_features.csv \
    --model output/model.joblib \
    --out output/test_matching_results.tsv

./package_submission.sh
echo "DONE"
