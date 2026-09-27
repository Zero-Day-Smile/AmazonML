"""
normalize.py
------------
Text normalization for business names and addresses.
Handles: transliteration, abbreviation expansion/contraction,
punctuation cleanup, and state name standardization.
"""

import re
import unicodedata
from unidecode import unidecode


# ── Business legal suffix normalisation ──────────────────────────────────────
# Always expand to the long form so "Ltd" == "Limited" == "Ltd."
LEGAL_SUFFIXES = {
    r"\bpvt\b":         "private",
    r"\bp\.v\.t\b":     "private",
    r"\bltd\b":         "limited",
    r"\bl\.t\.d\b":     "limited",
    r"\bllp\b":         "llp",
    r"\bllc\b":         "llc",
    r"\binc\b":         "incorporated",
    r"\binc\.\b":       "incorporated",
    r"\bcorp\b":        "corporation",
    r"\bcorp\.\b":      "corporation",
    r"\bco\b":          "company",
    r"\bco\.\b":        "company",
    r"\bpte\b":         "private",
    r"\bpte\.\b":       "private",
    r"\bplc\b":         "plc",
    r"\bgmbh\b":        "gmbh",
    r"\bintl\b":        "international",
    r"\bmfg\b":         "manufacturing",
    r"\bsvcs\b":        "services",
    r"\bsvc\b":         "service",
    r"\bassoc\b":       "associates",
    r"\bgrp\b":         "group",
    r"\bmgmt\b":        "management",
    r"\btech\b":        "technology",   # careful — tech is often a real word
    r"\bsol\b":         "solutions",
    r"\bdba\b":         "dba",          # keep dba — it's informative
    r"\bt/a\b":         "dba",          # trading as → dba
    r"\bta\b":          "dba",
}

# ── Address component normalisation ──────────────────────────────────────────
# Only expand unambiguous abbreviations (3+ chars or very clear context).
# Avoid 2-letter tokens that collide with common English words or state codes.
ADDR_TOKENS = {
    # Street types — only safe ones
    r"\bstr\b":         "street",
    r"\bave\b":         "avenue",
    r"\bblvd\b":        "boulevard",
    r"\bcir\b":         "circle",
    r"\bhwy\b":         "highway",
    r"\bfwy\b":         "freeway",
    r"\bpkwy\b":        "parkway",
    r"\bflr\b":         "floor",
    r"\bapt\b":         "apartment",
    r"\bste\b":         "suite",
    r"\bbldg\b":        "building",
    # Number label normalisation
    r"\bno\.\s*([\d\-]+)": r"number \1",  # "No. 536-" → "number 536-"
    # Landmark noise — drop entirely so they don't confuse similarity
    r"\bnear\b":        "",
    r"\bopp\b":         "",
    r"\bbehind\b":      "",
    r"\badjacent\b":    "",
    r"\bpincode\b":     "",
    r"\bpin code\b":    "",
    r"\bpo box\b":      "",
    r"\bp\.o\. box\b":  "",
}

# ── US State name ↔ abbreviation ─────────────────────────────────────────────
# Only expand full state names to 2-letter codes.
# Short 2-letter codes in source data stay as-is.
# We intentionally skip states whose full name is <= 3 chars to avoid
# accidental substring replacements (e.g. "Iowa" is safe at 4 chars).
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy",
}

# Only match full multi-char state names (>=4 chars) to avoid collision with
# common words: e.g. "Indiana Street" — "indiana" is 7 chars, safe to match.
# We rely on the address tokens step having already run so "street" etc. are
# already expanded; state names appear at segment ends after commas.
_LONG_US_STATES = {k: v for k, v in US_STATES.items() if len(k) >= 4}
_STATE_PATTERN = re.compile(
    r"(?<![a-z])(" + "|".join(re.escape(s) for s in sorted(_LONG_US_STATES, key=len, reverse=True)) + r")(?![a-z])"
)

# Indian states
INDIA_STATES = {
    "maharashtra": "mh", "karnataka": "ka", "tamil nadu": "tn",
    "uttar pradesh": "up", "rajasthan": "rj", "gujarat": "gj",
    "west bengal": "wb", "madhya pradesh": "mp", "andhra pradesh": "ap",
    "telangana": "tg", "kerala": "kl", "punjab": "pb", "haryana": "hr",
    "bihar": "br", "odisha": "od", "jharkhand": "jh", "assam": "as",
    "chhattisgarh": "cg", "himachal pradesh": "hp", "uttarakhand": "uk",
    "delhi": "dl", "goa": "ga",
}

_INDIA_PATTERN = re.compile(
    r"(?<![a-z])(" + "|".join(re.escape(s) for s in sorted(INDIA_STATES, key=len, reverse=True)) + r")(?![a-z])"
)


# ── Noise tokens to strip entirely ───────────────────────────────────────────
# Brackets around tokens, leading/trailing punctuation noise
_BRACKET_NOISE = re.compile(r"[\[\](){}<>]")
_MULTI_SPACE   = re.compile(r"\s{2,}")
_NON_ALPHANUM  = re.compile(r"[^a-z0-9\s]")


def _transliterate(text: str) -> str:
    """
    Convert any non-ASCII script (Devanagari, Kannada, Arabic, etc.)
    to closest ASCII equivalent via unidecode, then unicode normalise.
    """
    # NFKC first: decomposes ligatures, fullwidth chars, etc.
    text = unicodedata.normalize("NFKC", text)
    # unidecode handles the rest (Kannada, Hindi, Arabic → latin)
    return unidecode(text)


def normalize_name(raw: str) -> str:
    """
    Normalise a business name.
    Returns a clean, lowercase, space-separated token string.
    """
    if not raw or not isinstance(raw, str):
        return ""

    s = _transliterate(raw).lower()

    # Remove bracket noise (often wraps legal forms: "[co]", "(pvt)")
    s = _BRACKET_NOISE.sub(" ", s)

    # Expand legal suffixes
    for pattern, replacement in LEGAL_SUFFIXES.items():
        s = re.sub(pattern, replacement, s)

    # Drop anything that's not alphanumeric or whitespace
    s = _NON_ALPHANUM.sub(" ", s)

    # Collapse whitespace
    s = _MULTI_SPACE.sub(" ", s).strip()

    return s


def normalize_address(raw: str) -> str:
    """
    Normalise a business address.
    Returns a clean, lowercase, space-separated token string.
    """
    if not raw or not isinstance(raw, str):
        return ""

    s = _transliterate(raw).lower()

    # Collapse PO Box patterns before the rest
    s = re.sub(r"p\.?\s*o\.?\s*box\s*\d+", "", s)

    # Expand address tokens
    for pattern, replacement in ADDR_TOKENS.items():
        s = re.sub(pattern, " " + replacement + " ", s)

    # State normalisation (longest match first to avoid "new" matching early)
    s = _STATE_PATTERN.sub(lambda m: US_STATES[m.group(1)], s)
    s = _INDIA_PATTERN.sub(lambda m: INDIA_STATES[m.group(1)], s)

    # Drop non-alphanumeric chars
    s = _NON_ALPHANUM.sub(" ", s)

    # Collapse
    s = _MULTI_SPACE.sub(" ", s).strip()

    return s


def normalize_record(entity_id: str, name: str, address: str, country: str) -> dict:
    """
    Normalise a single record. Returns a dict with both raw and normalised fields.
    """
    norm_name    = normalize_name(name)
    norm_address = normalize_address(address)
    # For blocking we concatenate name+address into a single text blob
    combined     = (norm_name + " " + norm_address).strip()
    return {
        "entity_id":   entity_id,
        "country":     country.strip().lower() if country else "",
        "norm_name":   norm_name,
        "norm_addr":   norm_address,
        "combined":    combined,
    }


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("S1-001", "Zander Blue Co",                    "9236 Meadowmont View Drive, Charlotte, NC",            "US"),
        ("S2-001", "ZANDER BLUE [CO]",                  "9238 MEADOWMONT VIEW DRIVE, CHRALOTTE CITY, NC",       "US"),
        ("S3-001", "ZB",                                "9236 Meadowmont View Drive, Charlotte, NC",            "US"),
        ("S1-002", "Eye Physicians of Elmhurst",        "602 Indiana Street, Elmhurst, IL",                     "US"),
        ("S2-002", "Eye Physicians of Elmhurst Corporation", "602 INDIANA SAINT, ELMHURST, IL",                 "US"),
        ("S3-002", "Eye-Physicians of Elmhurst",        "02 Indiana Street, PO Box 260, Elmhurst, Illinois",    "US"),
        ("S1-003", "Gold Projects Pvt Ltd",             "Flat No.A-1101, 11Th Floor, Advaitha Aksha Apartment, Bangalore, Karnataka", "India"),
        ("S2-003", "ಗೋಲ್ಡ್ ಪ್ರಾಜೆಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "ಕರ್ನಾಟಕ, BANGALORE SOUTH, 11TH FLOOR, ADVAITHA AKSHA", "India"),
        ("S1-004", "Services Tycon Connect Partners",   "Shop No. G-01-C, Pragati Nagar, Ajmer, Rajasthan",     "India"),
        ("S2-004", "SERVICES TÉTONC CONNECT PARTNERS",  "",                                                     "India"),
    ]

    print(f"{'entity_id':<10} {'norm_name':<40} {'norm_addr':<55} {'country'}")
    print("-" * 120)
    for args in test_cases:
        r = normalize_record(*args)
        print(f"{r['entity_id']:<10} {r['norm_name'][:39]:<40} {r['norm_addr'][:54]:<55} {r['country']}")
