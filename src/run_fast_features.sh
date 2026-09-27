#!/bin/bash
set -e

echo "[2/4] Generating Pairwise Features using 10 Bash-level processes..."

# Split candidates into chunks (excluding header)
mkdir -p scratch_feats
tail -n +2 output/test_candidate_pairs.tsv > scratch_feats/cands_noheader.tsv
split -l 175000 scratch_feats/cands_noheader.tsv scratch_feats/part_

# Add headers back to parts
head -n 1 output/test_candidate_pairs.tsv > scratch_feats/header.tsv
for f in scratch_feats/part_*; do
    cat scratch_feats/header.tsv "$f" > "${f}_tmp"
    mv "${f}_tmp" "$f"
done

# Run fast_features.py on each part in parallel
TARGETS="dataset/test/test_source2.tsv,dataset/test/test_source3.tsv"

i=0
for f in scratch_feats/part_*; do
    # Only print header for the first chunk
    HEADER="0"
    if [ $i -eq 0 ]; then
        HEADER="1"
    fi
    
    python3 fast_features.py \
        --s1 dataset/test/test_source1.tsv \
        --targets "$TARGETS" \
        --candidates "$f" \
        --out "${f}_out.csv" \
        --header "$HEADER" &
        
    i=$((i+1))
done

echo "Waiting for all 10 processes to finish..."
wait

echo "Merging results..."
cat scratch_feats/part_*_out.csv > output/test_features.csv

echo "Cleaning up..."
rm -rf scratch_feats

echo "Done! Running prediction..."
python3 predict.py \
    --s1 dataset/test/test_source1.tsv \
    --features output/test_features.csv \
    --model output/model.joblib \
    --out output/test_matching_results.tsv

echo "Zipping..."
cd output && zip submission.zip test_matching_results.tsv test_candidate_pairs.tsv
echo "ALL DONE!"
