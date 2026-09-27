import argparse
import collections
import csv
import re
import sys
import time

csv.field_size_limit(sys.maxsize)

_MULTI_SPACE  = re.compile(r"\s{2,}")
_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")

def _norm_name(s):
    if not s: return ""
    s = s.lower()
    s = _NON_ALPHANUM.sub(" ", s)
    return _MULTI_SPACE.sub(" ", s).strip()

def _norm_country(s):
    return (s or "").strip().lower()

def load_source(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            records.append({
                "entity_id": row["entity_id"],
                "norm_name": _norm_name(row.get("business_name", "")),
                "country":   _norm_country(row.get("country", "")),
            })
    return records

def run_blocking(s1_path, s2_path, s3_path, out_path, top_k=30):
    t0 = time.time()

    print("Loading sources...")
    s1 = load_source(s1_path)
    targets = load_source(s2_path) + load_source(s3_path)
    
    print("Building indices...")
    name_exact = collections.defaultdict(list)
    word_index = collections.defaultdict(list)
    
    for i, r in enumerate(targets):
        c = r["country"]
        nn = r["norm_name"]
        if nn:
            name_exact[(nn, c)].append(i)
        
        # Unique words only per document
        for w in set(nn.split()):
            if len(w) >= 3: # Skip short words
                word_index[(w, c)].append(i)
                
    # Filter common words (max_df equivalent)
    print("Filtering common words...")
    filtered_word_index = {}
    for k, v in word_index.items():
        if len(v) < 10_000: # Max frequency
            filtered_word_index[k] = v
            
    word_index = filtered_word_index
            
    print("Generating candidates...")
    target_ids = [r["entity_id"] for r in targets]
    
    import os
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    
    total_candidates = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        
        for i, rec in enumerate(s1):
            s1_country = rec["country"]
            nn = rec["norm_name"]
            
            cand_scores = collections.defaultdict(int)
            
            # Exact match (infinite score)
            for idx in name_exact.get((nn, s1_country), []):
                cand_scores[idx] += 1000
                
            # Word match
            for w in set(nn.split()):
                if len(w) >= 3:
                    for idx in word_index.get((w, s1_country), []):
                        cand_scores[idx] += 1
                        
            # Get top K
            if len(cand_scores) > top_k:
                top_items = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                best_cands = [target_ids[idx] for idx, score in top_items]
            else:
                best_cands = [target_ids[idx] for idx in cand_scores]
                
            total_candidates += len(best_cands)
            writer.writerow([rec["entity_id"], ",".join(best_cands)])
            
            if i % 100_000 == 0:
                print(f"  Processed {i:,} queries...", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. Candidates written to {out_path}")
    print(f"Total candidate pairs: {total_candidates:,}")
    print(f"Avg candidates per S1: {total_candidates/len(s1):.1f}")
    print(f"Elapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1",   required=True)
    ap.add_argument("--s2",   required=True)
    ap.add_argument("--s3",   required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    run_blocking(args.s1, args.s2, args.s3, args.out, args.topk)
