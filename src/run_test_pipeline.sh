#!/bin/bash
# -----------------------------------------------------------------------------
# RUN FULL INFERENCE PIPELINE FOR SUBMISSION
# -----------------------------------------------------------------------------
set -e

echo "====================================================="
echo "  AMAZON ML CHALLENGE 2026: INFERENCE PIPELINE"
echo "====================================================="

# 1. Blocking / Candidate Generation
# 1. Blocking / Candidate Generation
# echo "[1/4] Running Candidate Generation (Inverted Index)..."
# python3 fast_block.py \
#     --s1 dataset/test/test_source1.tsv \
#     --s2 dataset/test/test_source2.tsv \
#     --s3 dataset/test/test_source3.tsv \
#     --out output/test_candidate_pairs.tsv \
#     --topk 30

# 2. Feature Engineering
echo "[2/4] Generating Pairwise Features..."
python3 features.py \
    --s1 dataset/test/test_source1.tsv \
    --s2 dataset/test/test_source2.tsv \
    --s3 dataset/test/test_source3.tsv \
    --candidates output/test_candidate_pairs.tsv \
    --out output/test_features.csv

# 3. Model Inference & Thresholding
echo "[3/4] Running Model Inference & Formatting Submission..."
python3 predict.py \
    --s1 dataset/test/test_source1.tsv \
    --features output/test_features.csv \
    --model output/model.joblib \
    --out output/test_matching_results.tsv

# 4. Validation & Packaging
echo "[4/4] Validating outputs against hackathon rules..."
python3 utils/validate_submission.py \
    --candidates output/test_candidate_pairs.tsv \
    --matches output/test_matching_results.tsv

echo "====================================================="
echo "✅ PIPELINE COMPLETE!"
echo "Your submission files are ready in the output/ folder."
echo "- output/test_matching_results.tsv"
echo "- output/test_candidate_pairs.tsv"
echo "Zip these two files together and upload to Unstop!"
echo "====================================================="

# 5. Zip for submission
echo 'Zipping submission...'
cd output && zip submission.zip test_matching_results.tsv test_candidate_pairs.tsv
echo 'Created output/submission.zip'
