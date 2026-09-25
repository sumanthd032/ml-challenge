"""Rule-based transliteration of Indic (Brahmic) scripts to simplified Latin.

All nine major Indic scripts in Unicode (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
Telugu, Kannada, Malayalam) share the ISCII-derived block layout: the same offset inside each
128-codepoint block encodes the same phonetic letter. We therefore map every Indic codepoint to its
offset and transliterate with one table. Output is a *simplified* romanization tuned for fuzzy
matching against English spellings (long vowels collapsed, retroflex/dental merged, ph->f).

No external data or service is used; this is a deterministic rule table.
"""
import re

_BLOCK_START, _BLOCK_END = 0x0900, 0x0D7F
# Scripts that normally delete the word-final inherent vowel ("राम" -> "ram", not "rama").
_SCHWA_DELETING = {0x0900, 0x0980, 0x0A00, 0x0A80}  # Devanagari, Bengali, Gurmukhi, Gujarati

_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
           0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
         0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
         0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b",
         0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
         0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
         0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
           0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x57: "au",
           0x62: "li", 0x63: "li"}
_NUKTA = {"j": "z", "jh": "z", "d": "r", "dh": "rh", "k": "q", "ph": "f", "f": "f"}
_VIRAMA = 0x4D
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}
# Malayalam chillu letters and Bengali khanda-ta are standalone dead consonants.
_SPECIAL = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k", 0x09CE: "t",
            0x0B83: "h"}


def is_indic_char(ch: str) -> bool:
    return _BLOCK_START <= ord(ch) <= _BLOCK_END


def has_indic(s: str) -> bool:
    return any(_BLOCK_START <= ord(c) <= _BLOCK_END for c in s)


def translit_word(word: str) -> str:
    """Transliterate one Indic-script word to simplified Latin. Non-Indic chars pass through."""
    out = []
    pending_cons = False   # last emitted item was a consonant still carrying its inherent 'a'
    block = None
    for ch in word:
        cp = ord(ch)
        if cp in (0x200C, 0x200D):          # zero-width (non-)joiner
            continue
        if cp in _SPECIAL:
            if pending_cons:
                out.append("a")
            out.append(_SPECIAL[cp]); pending_cons = False
            continue
        if not (_BLOCK_START <= cp <= _BLOCK_END):
            if pending_cons:
                out.append("a"); pending_cons = False
            out.append(ch)
            continue
        block = cp & ~0x7F
        off = cp - block
        if off in _CONS:
            if pending_cons:
                out.append("a")
            out.append(_CONS[off]); pending_cons = True
        elif off in _MATRAS:
            out.append(_MATRAS[off]); pending_cons = False
        elif off == _VIRAMA:
            pending_cons = False
        elif off in _VOWELS:
            if pending_cons:
                out.append("a")
            out.append(_VOWELS[off]); pending_cons = False
        elif off in _NASAL:
            if pending_cons:
                out.append("a"); pending_cons = False
            out.append(_NASAL[off])
        elif off == 0x3C and out:            # nukta modifies the previous consonant (ज़ -> z, ड़ -> r)
            out[-1] = _NUKTA.get(out[-1], out[-1])
        elif 0x66 <= off <= 0x6F:            # native digits
            if pending_cons:
                out.append("a"); pending_cons = False
            out.append(str(off - 0x66))
        # nukta (0x3C), avagraha, accents etc. are ignored
    if pending_cons and block not in _SCHWA_DELETING:
        out.append("a")
    s = "".join(out)
    s = re.sub(r"([aeiou])\1+", r"\1", s)    # collapse long vowels: aa -> a
    return s


def translit(text: str) -> str:
    """Transliterate every whitespace-separated word of `text` that contains Indic characters."""
    if not has_indic(text):
        return text
    return " ".join(translit_word(w) if has_indic(w) else w for w in text.split())


if __name__ == "__main__":
    for t in ["वन टेक्नोलॉजीज प्रा. लि.", "राम मार्केटिंग प्राइवेट लिमिटेड", "વિઝન પાવર પ્રાઇવેટ લિમિટેડ",
              "బిగ్ డిజిటల్ ఇన్‌ఫ్రా ప్రైవేట్ లిమిటెడ్", "महाराष्ट्र", "ಕರ್ನಾಟಕ", "పశ్చిమ", "তামিলনাড়ু",
              "தமிழ்நாடு", "ଓଡ଼ିଶା", "കേരളം"]:
        print(t, "->", translit(t))
