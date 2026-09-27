import argparse
import csv
import sys
from rapidfuzz.distance import JaroWinkler, Levenshtein

csv.field_size_limit(sys.maxsize)

def load_source(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # FIX: Handle BOTH 'address' and 'business_address'
            addr = row.get("business_address", row.get("address", ""))
            d[row["entity_id"]] = {
                "norm_name": str(row.get("business_name", "")).lower().strip(),
                "norm_addr": str(addr).lower().strip(),
                "country": str(row.get("country", "")).lower().strip()
            }
    return d

def run_features(s1_path, targets_path, candidates_path, out_path, print_header):
    s1 = load_source(s1_path)
    
    # Load targets (could be just one file or pre-merged, assuming caller passes S2/S3 merged or something. We'll load both from a single dir if needed. Wait, we can pass both via loop)
    targets = {}
    for t_path in targets_path.split(","):
        targets.update(load_source(t_path))
        
    cands = {}
    with open(candidates_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("candidate_entity_ids"):
                cands[row["source1_entity_id"]] = row["candidate_entity_ids"].split(",")

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        if print_header == "1":
            writer.writerow([
                "source1_entity_id", "candidate_entity_id",
                "name_exact_match", "addr_exact_match", "country_match",
                "name_levenshtein", "name_jaro_winkler", "addr_levenshtein",
                "name_length_diff", "addr_length_diff"
            ])
            
        for s1_id, c_ids in cands.items():
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--header", required=True)
    args = ap.parse_args()

    run_features(args.s1, args.targets, args.candidates, args.out, args.header)
