"""Normalisasi teks Bahasa Indonesia untuk training, inferensi, dan scoring."""
import re
import unicodedata

try:
    from num2words import num2words
except Exception:  # pragma: no cover
    num2words = None

ABBR = {
    "yg": "yang", "dgn": "dengan", "tdk": "tidak", "utk": "untuk", "dll": "dan lain lain",
    "dsb": "dan sebagainya", "dst": "dan seterusnya", "krn": "karena", "jgn": "jangan",
    "sdh": "sudah", "blm": "belum", "tp": "tetapi", "sbg": "sebagai", "spt": "seperti",
    "dr": "dari", "pd": "pada", "thd": "terhadap", "mis": "misalnya", "hlm": "halaman",
    "tsb": "tersebut", "ttg": "tentang", "kpd": "kepada", "bpk": "bapak", "sdr": "saudara",
}
UNITS = {
    "kg": "kilogram", "km": "kilometer", "cm": "sentimeter", "mm": "milimeter",
    "gb": "gigabyte", "mb": "megabyte", "kb": "kilobyte", "jt": "juta", "rb": "ribu",
}
_ID_DIGITS = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]


def _int_to_words(n: int) -> str:
    if num2words is not None:
        try:
            return num2words(n, lang="id")
        except Exception:
            pass
    return " ".join(_ID_DIGITS[int(d)] for d in str(n))


def _num_repl(m: re.Match) -> str:
    s = m.group(0)
    if "," in s:
        a, b = s.split(",", 1)
        a = a.replace(".", "")
        return f"{_int_to_words(int(a))} koma {' '.join(_ID_DIGITS[int(d)] for d in b)}"
    s = s.replace(".", "")
    if len(s) > 1 and s.startswith("0"):
        return " ".join(_ID_DIGITS[int(d)] for d in s)
    return _int_to_words(int(s))


def normalize_id(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower().strip()
    t = re.sub(r"\[/?en\]", " ", t, flags=re.I)
    t = t.replace("’", "'").replace("`", "'").replace("“", " ").replace("”", " ").replace('"', " ")
    t = re.sub(r"rp\.?\s*(\d[\d\.,]*)", lambda m: m.group(1) + " rupiah", t)
    t = t.replace("%", " persen").replace("&", " dan ").replace("+", " tambah ").replace("=", " sama dengan ")
    t = t.replace("@", " at ").replace("/", " per ")
    t = re.sub(r"[–—−]", " ", t)
    t = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " sampai ", t)
    t = re.sub(r"\d+(?:\.\d{3})*(?:,\d+)?|\d+", _num_repl, t)
    words = []
    for w in re.findall(r"[a-z']+|[,.?!;:]", t):
        if w in (";", ":"):
            w = ","
        words.append(UNITS.get(w, ABBR.get(w, w)) if w.isalpha() else w)
    t = " ".join(words)
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    t = re.sub(r"([,.?!])\1+", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    return t


def for_scoring(text: str) -> str:
    t = normalize_id(text)
    t = re.sub(r"[^a-z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
