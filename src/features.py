import argparse
import csv
import sys
from rapidfuzz.distance import JaroWinkler, Levenshtein
import time

csv.field_size_limit(sys.maxsize)

def load_source(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            addr = row.get("business_address", row.get("address", ""))
            d[row["entity_id"]] = {
                "norm_name": str(row.get("business_name", "")).lower().strip(),
                "norm_addr": str(addr).lower().strip(),
                "country": str(row.get("country", "")).lower().strip()
            }
    return d

def load_candidates(path):
    c = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cands = row.get("candidate_entity_ids", "")
            if cands:
                c[row["source1_entity_id"]] = cands.split(",")
    return c

def run_features(s1_path, s2_path, s3_path, candidates_path, out_path):
    t0 = time.time()
    
    print("Loading raw data for feature extraction...")
    s1 = load_source(s1_path)
    targets = load_source(s2_path)
    targets.update(load_source(s3_path))
    
    candidates = load_candidates(candidates_path)
    
    print("Generating features for candidate pairs...")
    
    header = [
        "source1_entity_id", "candidate_entity_id",
        "name_exact_match", "addr_exact_match", "country_match",
        "name_levenshtein", "name_jaro_winkler", "addr_levenshtein",
        "name_length_diff", "addr_length_diff"
    ]
    
    total_processed = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        
        for s1_id, c_ids in candidates.items():
            s1_row = s1.get(s1_id)
            if not s1_row: continue
            
            s1_n = s1_row["norm_name"]
            s1_a = s1_row["norm_addr"]
            s1_c = s1_row["country"]
            
            for c_id in c_ids:
                c_row = targets.get(c_id)
                if not c_row: continue
                
                c_n = c_row["norm_name"]
                c_a = c_row["norm_addr"]
                c_c = c_row["country"]
                
                name_exact_match = 1 if s1_n and c_n and s1_n == c_n else 0
                addr_exact_match = 1 if s1_a and c_a and s1_a == c_a else 0
                country_match = 1 if s1_c and c_c and s1_c == c_c else 0
                
                name_levenshtein = Levenshtein.normalized_similarity(s1_n, c_n) if s1_n and c_n else 0.0
                name_jaro_winkler = JaroWinkler.similarity(s1_n, c_n) if s1_n and c_n else 0.0
                addr_levenshtein = Levenshtein.normalized_similarity(s1_a, c_a) if s1_a and c_a else 0.0
                
                name_length_diff = abs(len(s1_n) - len(c_n))
                addr_length_diff = abs(len(s1_a) - len(c_a))
                
                writer.writerow([
                    s1_id, c_id,
                    name_exact_match, addr_exact_match, country_match,
                    name_levenshtein, name_jaro_winkler, addr_levenshtein,
                    name_length_diff, addr_length_diff
                ])
                total_processed += 1
                
                if total_processed % 100_000 == 0:
                    print(f"  Processed {total_processed:,} pairs...", flush=True)

    print(f"\nDone. Features written to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--s3", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_features(args.s1, args.s2, args.s3, args.candidates, args.out)
