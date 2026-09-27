"""
block.py
--------
Candidate generation (blocking) stage.

Strategy:
  1. Fast normalise all records (no unidecode — pure regex, fast enough).
  2. Build a TF-IDF matrix over S2+S3 using character 2-4 grams on
     the concatenated (norm_name + " " + norm_addr) text.
  3. For each S1 entity, retrieve top-K candidates via cosine similarity
     (batch sparse dot-product — vectorised, handles 10M rows fine).
  4. Union with an exact normalised-name lookup (catches cases the TF-IDF
     misses due to very short names or extreme abbreviations).
  5. Optionally filter by country — never pair entities from different
     countries (hard constraint, no exceptions seen in training data).
  6. Write candidate_pairs.tsv.

Usage:
    python3 block.py \\
        --s1  dataset/test/test_source1.tsv \\
        --s2  dataset/test/test_source2.tsv \\
        --s3  dataset/test/test_source3.tsv \\
        --out output/candidate_pairs.tsv \\
        --topk 30

For evaluation on training data, swap test paths for train paths and
provide --gt dataset/train/train_ground_truth.tsv to measure recall.
"""

import argparse
import collections
import csv
import re
import sys
import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

csv.field_size_limit(sys.maxsize)


# ── Fast normaliser (no unidecode — fast enough for blocking) ─────────────────
_MULTI_SPACE  = re.compile(r"\s{2,}")
_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")

_LEGAL = {
    "pvt": "private", "ltd": "limited", "corp": "corporation",
    "inc": "incorporated", "co": "company", "llc": "llc", "llp": "llp",
    "intl": "international", "assoc": "associates", "grp": "group",
    "mgmt": "management", "svcs": "services", "pte": "private",
    "mfg": "manufacturing", "svc": "service", "sol": "solutions",
}
_LEGAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _LEGAL) + r")\b"
)

_US_FULL = {
    "illinois": "il", "indiana": "in", "california": "ca", "texas": "tx",
    "florida": "fl", "georgia": "ga", "ohio": "oh", "michigan": "mi",
    "pennsylvania": "pa", "north carolina": "nc", "new york": "ny",
    "virginia": "va", "washington": "wa", "arizona": "az", "tennessee": "tn",
    "massachusetts": "ma", "maryland": "md", "wisconsin": "wi",
    "minnesota": "mn", "colorado": "co", "alabama": "al", "louisiana": "la",
    "kentucky": "ky", "oregon": "or", "oklahoma": "ok", "connecticut": "ct",
    "iowa": "ia", "mississippi": "ms", "arkansas": "ar", "kansas": "ks",
    "utah": "ut", "nevada": "nv", "new mexico": "nm", "nebraska": "ne",
    "west virginia": "wv", "idaho": "id", "hawaii": "hi", "maine": "me",
    "new hampshire": "nh", "rhode island": "ri", "montana": "mt",
    "delaware": "de", "south carolina": "sc", "north dakota": "nd",
    "south dakota": "sd", "alaska": "ak", "vermont": "vt", "wyoming": "wy",
    "missouri": "mo", "new jersey": "nj",
}
_STATE_RE = re.compile(
    r"(?<![a-z])(" + "|".join(
        re.escape(s) for s in sorted(_US_FULL, key=len, reverse=True)
    ) + r")(?![a-z])"
)

_INDIA_STATES = {
    "maharashtra": "mh", "karnataka": "ka", "tamil nadu": "tn",
    "uttar pradesh": "up", "rajasthan": "rj", "gujarat": "gj",
    "west bengal": "wb", "madhya pradesh": "mp", "andhra pradesh": "ap",
    "telangana": "tg", "kerala": "kl", "punjab": "pb", "haryana": "hr",
    "bihar": "br", "odisha": "od", "jharkhand": "jh", "assam": "as",
    "chhattisgarh": "cg", "himachal pradesh": "hp", "uttarakhand": "uk",
    "delhi": "dl",
}
_INDIA_RE = re.compile(
    r"(?<![a-z])(" + "|".join(
        re.escape(s) for s in sorted(_INDIA_STATES, key=len, reverse=True)
    ) + r")(?![a-z])"
)


def _norm_name(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _NON_ALPHANUM.sub(" ", s)
    s = _LEGAL_RE.sub(lambda m: _LEGAL[m.group(1)], s)
    return _MULTI_SPACE.sub(" ", s).strip()


def _norm_addr(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"p\.?\s*o\.?\s*box\s*[\d\w\-]*", "", s)  # strip PO Box
    s = _STATE_RE.sub(lambda m: _US_FULL[m.group(1)], s)
    s = _INDIA_RE.sub(lambda m: _INDIA_STATES[m.group(1)], s)
    s = _NON_ALPHANUM.sub(" ", s)
    return _MULTI_SPACE.sub(" ", s).strip()


def _norm_country(s: str) -> str:
    return (s or "").strip().lower()


def load_source(path: str):
    """
    Returns list of dicts with keys:
      entity_id, norm_name, norm_addr, combined, country
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            nn = _norm_name(row.get("business_name", ""))
            na = _norm_addr(row.get("business_address", ""))
            records.append({
                "entity_id": row["entity_id"],
                "norm_name": nn,
                "norm_addr": na,
                "combined":  (nn + " " + na).strip(),
                "country":   _norm_country(row.get("country", "")),
            })
    return records


# ── Blocking ──────────────────────────────────────────────────────────────────
import random

def build_tfidf(texts, max_features=100_000):
    """Fit TF-IDF on a sample of texts to save memory. Returns vectorizer."""
    print(f"  Sampling 500k texts for TF-IDF fit (out of {len(texts):,})...")
    sample_texts = random.sample(texts, min(500_000, len(texts)))
    
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )
    vec.fit(sample_texts)
    return vec


def retrieve_topk_chunked(s1_texts, vec, target_texts, top_k=30, q_chunk_size=5000, t_chunk_size=50_000):
    """
    Computes cosine similarity between queries and targets using double-chunking.
    Uses pure CSR sparse matrix extraction to avoid any dense array conversions.
    """
    n_queries = len(s1_texts)
    n_targets = len(target_texts)
    
    print("  Pre-transforming all S1 queries...")
    q_mats_norm = []
    for q_start in range(0, n_queries, q_chunk_size):
        q_end = min(q_start + q_chunk_size, n_queries)
        q_mat = vec.transform(s1_texts[q_start:q_end])
        q_norms = np.asarray(q_mat.power(2).sum(axis=1)).flatten() ** 0.5
        q_norms[q_norms == 0] = 1.0
        q_mats_norm.append(q_mat.multiply(1.0 / q_norms[:, None]))
        
    best_scores = np.full((n_queries, top_k), -1.0, dtype=np.float32)
    best_indices = np.full((n_queries, top_k), -1, dtype=np.int32)
    
    for t_start in range(0, n_targets, t_chunk_size):
        t_end = min(t_start + t_chunk_size, n_targets)
        print(f"  Processing target chunk {t_start:,} to {t_end:,}...")
        
        tgt_chunk = vec.transform(target_texts[t_start:t_end])
        t_norms = np.asarray(tgt_chunk.power(2).sum(axis=1)).flatten() ** 0.5
        t_norms[t_norms == 0] = 1.0
        tgt_chunk_norm = tgt_chunk.multiply(1.0 / t_norms[:, None]).tocsr()
        
        for chunk_idx, q_start in enumerate(range(0, n_queries, q_chunk_size)):
            q_end = min(q_start + q_chunk_size, n_queries)
            q_mat_norm = q_mats_norm[chunk_idx]
            
            # Sparse dot product (extremely fast because of max_df)
            scores = (q_mat_norm @ tgt_chunk_norm.T)
            
            for local_q_idx in range(q_end - q_start):
                global_q_idx = q_start + local_q_idx
                
                # Extract non-zero elements for this specific query directly from CSR
                start_ptr = scores.indptr[local_q_idx]
                end_ptr = scores.indptr[local_q_idx+1]
                q_scores = scores.data[start_ptr:end_ptr]
                q_cols = scores.indices[start_ptr:end_ptr]
                
                if len(q_scores) == 0:
                    continue
                
                if len(q_scores) > top_k:
                    chunk_top_idx = np.argpartition(q_scores, -top_k)[-top_k:]
                else:
                    chunk_top_idx = np.arange(len(q_scores))
                    
                chunk_top_scores = q_scores[chunk_top_idx]
                chunk_top_cols = q_cols[chunk_top_idx] + t_start
                
                # Combine with running bests
                combined_scores = np.concatenate([best_scores[global_q_idx], chunk_top_scores])
                combined_indices = np.concatenate([best_indices[global_q_idx], chunk_top_cols])
                
                new_top_idx = np.argpartition(combined_scores, -top_k)[-top_k:]
                
                best_scores[global_q_idx] = combined_scores[new_top_idx]
                best_indices[global_q_idx] = combined_indices[new_top_idx]
                
    return best_indices.tolist()


def run_blocking(s1_path, s2_path, s3_path, out_path, top_k=30, gt_path=None):
    t0 = time.time()

    print("Loading sources...")
    s1 = load_source(s1_path)
    s2 = load_source(s2_path)
    s3 = load_source(s3_path)
    targets = s2 + s3
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  targets={len(targets):,}")

    # Build country lookup for targets so we can filter cross-country pairs
    target_country = [r["country"] for r in targets]
    target_id      = [r["entity_id"] for r in targets]

    # Exact normalised-name lookup per country bucket
    print("Building exact-name index...")
    name_exact = collections.defaultdict(list)   # (norm_name, country) → [idx in targets]
    for i, r in enumerate(targets):
        if r["norm_name"]:
            name_exact[(r["norm_name"], r["country"])].append(i)

    # TF-IDF index over combined text
    print("Fitting TF-IDF on targets...")
    target_texts = [r["combined"] if r["combined"] else r["norm_name"] for r in targets]
    vec = build_tfidf(target_texts)
    
    # Retrieve top-K for each S1
    print("Retrieving top-K candidates via chunked TF-IDF...")
    s1_texts = [r["combined"] if r["combined"] else r["norm_name"] for r in s1]
    tfidf_indices = retrieve_topk_chunked(s1_texts, vec, target_texts, top_k=top_k)

    # Build final candidate sets (TF-IDF + exact name, filtered by country)
    print("Merging candidates and writing output...")
    import os
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    total_candidates = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for i, rec in enumerate(s1):
            s1_country = rec["country"]
            cand_set   = set()

            # TF-IDF candidates, same country only
            for idx in tfidf_indices[i]:
                if idx != -1:
                    if target_country[idx] == s1_country or not s1_country:
                        cand_set.add(target_id[idx])

            # Exact name candidates
            for idx in name_exact.get((rec["norm_name"], s1_country), []):
                cand_set.add(target_id[idx])

            total_candidates += len(cand_set)
            writer.writerow([rec["entity_id"], ",".join(sorted(cand_set))])

    elapsed = time.time() - t0
    print(f"\nDone. Candidates written to {out_path}")
    print(f"Total candidate pairs: {total_candidates:,}")
    print(f"Avg candidates per S1: {total_candidates/len(s1):.1f}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Optional: evaluate recall if GT is provided
    if gt_path:
        _eval_recall(out_path, gt_path)


def _eval_recall(cand_path, gt_path):
    print("\n--- Blocking recall evaluation ---")
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids = [x.strip() for x in row["matched_entity_ids"].split(",")
                   if x.strip()] if row["matched_entity_ids"] else []
            gt[row["source1_entity_id"]] = set(ids)

    hits = 0
    total = 0
    with open(cand_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s1id = row["source1_entity_id"]
            true_set = gt.get(s1id, set())
            if not true_set:
                continue
            cands = set(x for x in row["candidate_entity_ids"].split(",") if x)
            hits  += len(cands & true_set)
            total += len(true_set)

    print(f"Blocking recall: {hits:,} / {total:,} = {hits/total*100:.2f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1",   required=True)
    ap.add_argument("--s2",   required=True)
    ap.add_argument("--s3",   required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--gt",   default=None, help="ground truth TSV for recall eval")
    args = ap.parse_args()

    run_blocking(args.s1, args.s2, args.s3, args.out, args.topk, args.gt)
