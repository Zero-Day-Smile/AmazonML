# Amazon ML Challenge 2026 - Business Entity Resolution

## Pipeline Execution
This pipeline is designed to be fully self-contained. It extracts candidate pairs using an inverted word index, calculates Jaro-Winkler and Levenshtein string distances, and predicts matches using a HistGradientBoostingClassifier.

To run the pipeline end-to-end on the test set:
```bash
./run_test_pipeline.sh
```

## Structure
- `fast_block.py`: O(N) word-level inverted index for candidate generation (Blocking).
- `features.py`: Exact and fuzzy string matching across candidate pairs.
- `train.py`: Trains the ML model and tunes the F0.5 threshold.
- `predict.py`: Runs inference and strictly formats the output TSV to include all singletons.
