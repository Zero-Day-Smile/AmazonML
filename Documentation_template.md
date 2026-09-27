# Methodology

## 1. Candidate Generation / Blocking
Due to the sheer scale of comparing 1.7M queries against 10M targets, standard TF-IDF cosine similarity matrices caused massive Out-Of-Memory errors. We migrated to a pure Python O(N) inverted word index. 
We indexed all target entities by exact name and by unique tokens (ignoring words that appeared in >10,000 targets). For each test query, we retrieved targets that shared exact matches or rare tokens. This successfully reduced the trillions of combinations down to 42.2 million highly probable candidate pairs while executing in under 20 minutes.

## 2. Feature Engineering
For all 42.2 million pairs, we extracted standard string similarity metrics using the `rapidfuzz` C++ backend:
- Name Exact Match
- Address Exact Match
- Country Match
- Name Levenshtein Normalized Similarity
- Name Jaro-Winkler Similarity
- Address Levenshtein Normalized Similarity
- Name Length Difference
- Address Length Difference

## 3. Model Architecture
Given the strict hackathon constraints (Open Source, <8B parameters) and the tabular nature of the data, we utilized `scikit-learn`'s `HistGradientBoostingClassifier`. This model natively handles missing values and executes extremely fast without the need for external C++ dependencies (unlike XGBoost).

## 4. Threshold Tuning for F_0.5
Because the evaluation metric (F_0.5) penalizes false merges (false positives) twice as heavily as missed matches (false negatives), a standard 0.5 probability threshold is highly suboptimal.
During validation, we plotted the Precision-Recall curve and identified that a strict threshold of **0.9387** maximized the F_0.5 score by guaranteeing high precision.

## 5. Singletons
Entities with zero candidates, or entities where all candidates scored below 0.9387, are natively preserved in our pipeline as singletons with blank `matched_entity_ids`, successfully capturing the 1.0 score for non-matching businesses.
