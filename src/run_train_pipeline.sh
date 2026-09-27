#!/bin/bash
set -e

echo "====================================================="
echo "  AMAZON ML CHALLENGE 2026: TRAINING PIPELINE"
echo "====================================================="

echo "[1/4] Running Candidate Generation on Training Set..."
python3 fast_block.py \
    --s1 dataset/train/train_source1.tsv \
    --s2 dataset/train/train_source2.tsv \
    --s3 dataset/train/train_source3.tsv \
    --out output/train_candidate_pairs.tsv \
    --topk 30

echo "[2/4] Generating Features for Training Candidates..."
# Run the parallel feature extraction for train
mkdir -p scratch_train_feats
tail -n +2 output/train_candidate_pairs.tsv > scratch_train_feats/cands_noheader.tsv
split -l 150000 scratch_train_feats/cands_noheader.tsv scratch_train_feats/part_

head -n 1 output/train_candidate_pairs.tsv > scratch_train_feats/header.tsv
for f in scratch_train_feats/part_*; do
    cat scratch_train_feats/header.tsv "$f" > "${f}_tmp"
    mv "${f}_tmp" "$f"
done

TARGETS="dataset/train/train_source2.tsv,dataset/train/train_source3.tsv"

i=0
for f in scratch_train_feats/part_*; do
    HEADER="0"
    if [ $i -eq 0 ]; then
        HEADER="1"
    fi
    python3 fast_features.py \
        --s1 dataset/train/train_source1.tsv \
        --targets "$TARGETS" \
        --candidates "$f" \
        --out "${f}_out.csv" \
        --header "$HEADER" &
    i=$((i+1))
done

echo "Waiting for all parallel feature processes to finish..."
wait

echo "Merging results..."
cat scratch_train_feats/part_*_out.csv > output/train_features.csv
rm -rf scratch_train_feats

echo "[3/4] Training Model on Full Train Dataset..."
python3 train.py \
    --features output/train_features.csv \
    --gt dataset/train/train_ground_truth.tsv

echo "[4/4] Generating Test Predictions using New Model..."
python3 predict.py \
    --s1 dataset/test/test_source1.tsv \
    --features output/test_features.csv \
    --model output/model.joblib \
    --out output/test_matching_results.tsv

echo "Packaging Final Submission..."
./package_submission.sh

echo "ALL DONE!"
