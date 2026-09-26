"""Text normalization for business names and addresses.

Everything here is deterministic, rule-based domain knowledge (abbreviation tables, legal-form
lists, state abbreviations). No external lookup is performed.

Main entry points:
    norm_name(raw)    -> dict(name, core, concat, alt)
    norm_address(raw) -> dict(addr, toks, nums, state)
"""
import re
import unicodedata

from translit import translit, has_indic

# ----------------------------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------------------------
_NON_WORD = re.compile(r"[^\wऀ-ൿ]+")   # keep letters, digits, and Indic marks
_LATIN_ACCENT_RANGE = re.compile(r"[À-ɏḀ-ỿ]")


def strip_latin_accents(s: str) -> str:
    """Remove diacritics from Latin letters only (Indic vowel signs are combining marks too, keep them)."""
    if not _LATIN_ACCENT_RANGE.search(s):
        return s
    out = []
    for ch in s:
        if "À" <= ch <= "ɏ" or "Ḁ" <= ch <= "ỿ":
            d = unicodedata.normalize("NFKD", ch)
            out.append("".join(c for c in d if not unicodedata.combining(c)))
        else:
            out.append(ch)
    s = "".join(out)
    return s.replace("ß", "ss").replace("æ", "ae").replace("œ", "oe").replace("ø", "o").replace("ł", "l")


def basic_clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = strip_latin_accents(s)
    return s


# leetspeak / OCR confusions, applied only inside tokens mixing letters and digits
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def fix_leet(tok: str) -> str:
    if tok.isalpha() or tok.isdigit():
        return tok
    letters = sum(c.isalpha() for c in tok)
    if letters >= 2 and letters >= len(tok) - 2:      # mostly letters with 1-2 digits: H0using, Va1enzuela
        return tok.translate(_LEET)
    return tok


# ----------------------------------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------------------------------
# canonical form for legal / organisational suffix variants (EN, IN, FR)
LEGAL_CANON = {
    "pvt": "private", "pvt.": "private", "prv": "private", "pte": "private", "private": "private",
    "ltd": "limited", "limited": "limited", "ltda": "limited",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cos": "co", "compagnie": "co", "cie": "co",
    "llc": "llc", "l.l.c": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "pa": "pa", "ltee": "limited", "gmbh": "gmbh",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci", "sa": "sa", "snc": "snc",
    "ei": "ei", "selarl": "selarl", "scp": "scp", "scm": "scm", "sca": "sca", "gie": "gie",
    "ets": "etablissements", "etablissements": "etablissements", "tbk": "tbk", "opc": "opc",
    "holdings": "holdings", "group": "group", "groupe": "group",
    "praivet": "private", "piraivet": "private", "praibhet": "private", "privet": "private", "limitet": "limited",
}
# tokens that carry no identity (removed in the "core" name)
LEGAL_TOKENS = {"private", "limited", "inc", "corp", "co", "llc", "llp", "lp", "plc", "pllc", "pc", "pa",
                "gmbh", "sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "ei", "selarl", "scp", "scm",
                "sca", "gie", "etablissements", "tbk", "opc", "the", "and", "of", "et", "de", "du", "des",
                "la", "le", "les", "l", "d", "india", "m", "s", "ms",
                "shri", "sri", "shree", "smt", "dr", "mr", "mrs", "com", "www", "net", "org", "in", "id"}
_ALT_SPLIT = re.compile(r"\s(?:d/b/a|dba|t/a|a/k/a|aka|formerly(?: known as)?|f/k/a|trading as|doing business as)\s|\s\|\s",
                        re.I)
_NAME_JUNK = [
    (re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)", re.I), " "),     # (ID: 123)
    (re.compile(r"#\s*\d+"), " "),                               # #77599
    (re.compile(r"\bwww\."), " "),
    (re.compile(r"\.(com|net|org|in|co|biz|info|fr)\b"), " "),
    (re.compile(r"&"), " and "),
    (re.compile(r"\+"), " plus "),
    (re.compile(r"['`’]"), ""),                                  # gilda's -> gildas
]


def _load_indic_dict() -> dict:
    """Native-token -> English dictionary learned from train pairs (learn_dict.py); empty if absent."""
    import json
    import os
    from pathlib import Path
    p = Path(os.environ.get("BER_ARTIFACTS", Path(__file__).resolve().parents[3] / "artifacts")) / "indic_dict.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    return {}


INDIC_DICT = _load_indic_dict()


def map_indic(s: str) -> str:
    """Replace native-script words by their learned English translation, else rule-transliterate."""
    out = []
    for w in _NON_WORD.sub(" ", s).split():
        if has_indic(w):
            out.append(INDIC_DICT.get(w) or translit(w))
        else:
            out.append(w)
    return " ".join(out)


def _name_tokens(s: str) -> list:
    for pat, rep in _NAME_JUNK:
        s = pat.sub(rep, s)
    if has_indic(s):
        s = map_indic(s)
    toks = [fix_leet(t) for t in _NON_WORD.sub(" ", s).replace("_", " ").split()]
    return [LEGAL_CANON.get(t, t) for t in toks]


_DOTTED = re.compile(r"\b(?:[a-z]\.\s?){2,}(?:[a-z]\b\.?)?")


def _undot(s: str) -> str:
    """'s.a.r.l.' -> 'sarl', 'e.u.r.l' -> 'eurl', 's.a.' -> 'sa' (dotted acronyms / legal forms)."""
    return _DOTTED.sub(lambda m: re.sub(r"[.\s]", "", m.group()) + " ", s)


def norm_name(raw: str, fr_fix: bool = False, drop_country: bool = False) -> dict:
    """Return normalized name variants.

    name   : all canonical tokens (legal forms canonicalized)
    core   : identity tokens only (legal forms, honorifics, stop words removed)
    concat : core tokens joined without spaces (to match 'firstave.com'-style names)
    alt    : core of the alternative/trade name after dba/t/a/| if present, else ''
    fr_fix : France fix (D-019): dotted legal forms collapsed ('S.A.R.L.' was split into stray letters that
             stayed in the core). Off by default so India/US normalization is unchanged.
    drop_country: also drop 'france' from the core like 'india'. Tried and rejected (D-019): '(France)' is part of
             the name's identity (exact no-address copies lost their match, 'franceconcept.com' no longer matched).
    """
    s = basic_clean(raw)
    if fr_fix:
        s = _undot(s)
    parts = _ALT_SPLIT.split(f" {s} ")
    toks = _name_tokens(s)
    legal = LEGAL_TOKENS | {"france"} if drop_country else LEGAL_TOKENS
    core = [t for t in toks if t not in legal]
    if not core:
        core = toks
    alt = ""
    if len(parts) > 1:
        alt_toks = [t for t in _name_tokens(parts[-1]) if t not in legal]
        first = [t for t in _name_tokens(parts[0]) if t not in legal]
        # keep the part that is NOT the main core as the alternative; main core = first part
        core = first or core
        alt = " ".join(alt_toks)
    core_s = " ".join(core)
    return {"name": " ".join(toks), "core": core_s, "concat": "".join(core), "alt": alt}


# ----------------------------------------------------------------------------------------------
# Addresses
# ----------------------------------------------------------------------------------------------
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "tamilnadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "ladakh": "la", "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
    # transliterated native-script state names (output of translit.py)
    "maharashtr": "mh", "karnatak": "ka", "tamizhnatu": "tn", "tamilnatu": "tn", "tamilanadu": "tn",
    "telangan": "tg", "kerlam": "kl", "keralan": "kl", "keralam": "kl", "gujarat": "gj", "gujrat": "gj",
    "pashchim bangal": "wb", "pashchima bangal": "wb", "pashchim banga": "wb", "uttar pradesh": "up",
    "uttar prdesh": "up", "rajasthan": "rj", "dilli": "dl", "dili": "dl", "andhrapradesh": "ap",
    "andhra pradesh": "ap", "andhraprdesh": "ap", "madhya pradesh": "mp", "panjab": "pb", "bihar": "br",
    "hariyana": "hr", "odisha": "od", "asam": "as",
}
FR_REGIONS = {"hauts de france": "hdf", "nouvelle aquitaine": "naq", "pays de la loire": "pdl",
              "ile de france": "idf", "occitanie": "occ", "bretagne": "bre", "normandie": "nor",
              "grand est": "ges", "auvergne rhone alpes": "ara", "provence alpes cote d azur": "pac",
              "bourgogne franche comte": "bfc", "centre val de loire": "cvl", "corse": "cor"}
STATE_PHRASES = {}
for _d in (US_STATES, IN_STATES, FR_REGIONS):
    STATE_PHRASES.update(_d)
STATE_CODES = set(STATE_PHRASES.values())
# longest phrases first so "west virginia" wins over "virginia"
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATE_PHRASES), key=len, reverse=True)) + r")\b")

ADDR_CANON = {
    # street types (EN)
    "street": "st", "str": "st", "st": "st", "saint": "st", "road": "rd", "rd": "rd", "avenue": "ave",
    "av": "ave", "ave": "ave", "aven": "ave", "drive": "dr", "dr": "dr", "drv": "dr", "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "circle": "cir", "cir": "cir", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd",
    "bld": "blvd", "place": "pl", "pl": "pl", "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy",
    "hwy": "hwy", "trail": "trl", "trl": "trl", "terrace": "ter", "ter": "ter", "square": "sq", "sq": "sq",
    "way": "way", "cove": "cv", "cv": "cv", "point": "pt", "pt": "pt", "crossing": "xing", "loop": "loop",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "county": "co", "cnty": "co", "township": "twp",
    "twp": "twp", "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
    "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    # FR
    "rue": "rue", "r": "rue", "chemin": "ch", "che": "ch", "chem": "ch", "impasse": "imp", "imp": "imp",
    "allee": "all", "all": "all", "route": "rte", "rte": "rte", "quai": "quai", "cours": "crs", "crs": "crs",
    "residence": "res", "res": "res", "faubourg": "fg", "fbg": "fg", "ste": "ste", "sainte": "ste",
    # IN
    "nagar": "ngr", "ngr": "ngr", "marg": "marg", "colony": "col", "col": "col", "sector": "sec", "sec": "sec",
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "bengaluru": "bangalore",
    "gurugram": "gurgaon", "ahmadabad": "ahmedabad",
}
# tokens that are pure noise for matching (unit designators, filler words, null literals)
ADDR_STOP = {"null", "none", "na", "n", "a", "nan", "unit", "apt", "apartment", "suite", "ste_", "fl", "floor",
             "flr", "pmb", "po", "box", "no", "number", "h", "hno", "house", "door", "plot", "flat", "shop",
             "office", "bldg", "building", "near", "nr", "opp", "opposite", "behind", "beside", "c", "o",
             "the", "of", "and", "de", "du", "des", "la", "le", "les", "l", "d", "bis", "ter_", "numero",
             "cdp", "city", "region", "dist", "district", "tq", "taluka", "tal", "po_", "ps", "vill", "village",
             "at", "post", "via", "block", "blk", "room", "rm", "gf", "ff", "sf", "ground", "level",
             "lgf", "ugf", "tower", "wing", "phase", "ph"}
_NUM = re.compile(r"\d+")


# French departements -> region name (D-019). S1 records carry the region ("Hauts-de-France"), S2/S3 often the
# departement ("Nord", "Pas-de-Calais", "Gironde", "Loire-Atlantique"), so the state never matched.
FR_DEPTS = {d: "hauts de france" for d in ("nord", "pas de calais", "somme", "aisne", "oise")}
FR_DEPTS.update({d: "nouvelle aquitaine" for d in (
    "gironde", "landes", "dordogne", "lot et garonne", "pyrenees atlantiques", "charente", "charente maritime",
    "deux sevres", "vienne", "haute vienne", "creuse", "correze")})
FR_DEPTS.update({d: "pays de la loire" for d in ("loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee")})


def _fr_depts(raw: str) -> str:
    """Replace a comma component that is a whole departement name by its region name."""
    parts = raw.split(",")
    out = []
    for p in parts:
        k = re.sub(r"\s+", " ", _NON_WORD.sub(" ", basic_clean(p))).strip()
        out.append(" " + FR_DEPTS[k] if k in FR_DEPTS else p)
    return ",".join(out)


def norm_address(raw: str, fr_fix: bool = False) -> dict:
    """Return normalized address pieces.

    addr  : canonical token string (states replaced by codes, abbreviations canonicalized)
    toks  : set-like space-joined word tokens without numbers/states/stop words
    nums  : space-joined numbers (leading zeros stripped) in order of appearance
    state : state/region code if recognized ('' otherwise)
    fr_fix: France fixes (D-019): departements mapped to their region, and no 2-letter US/Indian state fallback
            (it read the articles 'de' / 'la' as Delaware / Louisiana). Off by default.
    """
    if fr_fix:
        raw = _fr_depts(raw)
    s = basic_clean(raw)
    if has_indic(s):
        s = translit(s)
    s = _NON_WORD.sub(" ", s.replace("_", " "))
    s = re.sub(r"\s+", " ", s).strip()
    state = ""
    m = list(_STATE_RE.finditer(s))
    if m:
        state = STATE_PHRASES[m[-1].group(1)]
        s = _STATE_RE.sub(lambda x: " st8_" + STATE_PHRASES[x.group(1)] + " ", s)
    toks, words, nums = [], [], []
    for t in s.split():
        if t.startswith("st8_"):
            toks.append(t)
            continue
        t = ADDR_CANON.get(t, t)
        for n in _NUM.findall(t):
            n = n.lstrip("0") or "0"
            nums.append(n)
        toks.append(t)
        if not t.isdigit() and t not in ADDR_STOP and not _NUM.fullmatch(t):
            alpha = re.sub(r"\d+", "", t) if any(c.isdigit() for c in t) and not re.fullmatch(r"\d+(st|nd|rd|th)", t) else t
            if len(alpha) >= 2 and alpha not in ADDR_STOP:
                words.append(alpha)
    # 2-letter state code at the end without full name (e.g. ", TX" or ", MH")
    if not state and not fr_fix:
        for t in reversed(toks[-3:]):
            if len(t) == 2 and t in STATE_CODES:
                state = t
                break
    words = [w for w in words if w != state and not w.startswith("st8_")]
    return {"addr": " ".join(toks), "toks": " ".join(words), "nums": " ".join(nums), "state": state}


if __name__ == "__main__":
    tests_n = ["Gilda's Pharmaceuticals Ltd | www.gildasph.com", "H0using Program", "Va1enzuela, Valenzuela, Vere,",
               "Irivantage t/a Valenzuela, Vere, DDS", "वन टेक्नोलॉजीज प्रा. लि.", "Lalwani Herbal Pite Limited #77599",
               "4220firstave.Com", "#vmuniholdings", "Anish Ltd Pvt [Healthcare]", "<< Team Ecole",
               "Supérieure Sante EURL", "Shri Supreme Consulting Private  (Limited)", "Gmk & S0ns",
               "Martin and Bdkc Design (ID: 7788)"]
    for t in tests_n:
        print(repr(t), "->", norm_name(t))
    tests_a = ["4809 HARRISON FERBY RD, HURLOCK, MD", "ASHBURN, 0020718 ADAMS MILL PLACE, VA",
               "Block C-303 C/o. Steelcast Limited Ruvapari Road, Bhavnagar, GJ",
               "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ",
               "300 Audubon Pkwy, # Unit 76, Syracuse, New York", "12 RUE de lOrne, 44800, Saint-Herblain, Pays de la Loire",
               "3B RUE DU PETIT VILLAGE, ST.-HERBLAIN, Pays de la Loire", "Plot No 17., NULL, Nagpur, MH",
               "H. No. 8-2-293/82/A/727 & 727/1/2, Hyderabad, తెలంగాణ", "HOMELAND AVE, NULL, NORMAN, OK", ""]
    for t in tests_a:
        print(repr(t), "->", norm_address(t))
