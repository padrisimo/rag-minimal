import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_SUFFIXES = (
    "amientos",
    "imientos",
    "amiento",
    "imiento",
    "aciones",
    "acion",
    "adores",
    "adora",
    "mente",
    "ando",
    "iendo",
    "ar",
    "er",
    "ir",
    "as",
    "es",
    "s",
    "a",
    "o",
)
_MIN_STEM = 4

STOPWORDS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes aqui asi aun aunque
    bajo bien cada como con contra cual cuales cuando de del desde donde dos
    durante e el ella ellas ellos en entre era eran eres es esa esas ese eso
    esta estaba estan estas este esto estos estoy fue fueron ha habia hacen
    hasta hay la las le les lo los mas me mi mis mientras mucho muy nada ni
    no nos nosotros o os otra otras otro otros para pero poco por porque que
    quien quienes se sea segun ser si sido sin sobre solo son su sus tambien
    tanto te tiene tienen toda todas todo todos tras tu tus un una unas uno
    unos usted ustedes va vamos van ver vez y ya yo
    a about after all also an and any are as at be because been before being
    but by can did do does doing for from further had has have having he her
    here hers him his how i if in into is it its itself just more most my
    no nor not now of off on once only or other our out over own same she
    should so some such than that the their them then there these they this
    those through to too under until up very was we were what when where
    which while who why will with would you your
    """.split()
)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            return word[: -len(suffix)]
    return word


def tokenize(text: str, ngram_max: int = 2) -> list[str]:
    words = [_stem(w) for w in _TOKEN_RE.findall(_fold(text)) if len(w) > 1 and w not in STOPWORDS]
    words = [w for w in words if w]
    grams: list[str] = list(words)
    if ngram_max >= 2:
        grams += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return grams