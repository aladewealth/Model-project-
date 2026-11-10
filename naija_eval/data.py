"""Load small, seeded slices of public Nigerian-language benchmarks.

Sources (all ungated, downloaded straight from their hosts):
  - AfriMMLU  (masakhane/afrimmlu)   multiple-choice knowledge, 4 options
  - AfriMGSM  (masakhane/afrimgsm)   grade-school math word problems
  - LAFAND-MT (masakhane-io/lafand-mt) news translation, English <-> hau/ibo/yor
"""
from __future__ import annotations

import ast
import csv
import io
import os
import random
import urllib.request

HF = "https://huggingface.co/datasets/masakhane/{name}/resolve/main/data/{lang}/test.tsv"
LAFAND = "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/text_files/en-{lang}/test.{side}"

LANGS = ["yor", "hau", "ibo"]
LANG_NAMES = {"yor": "Yoruba", "hau": "Hausa", "ibo": "Igbo", "eng": "English"}
CACHE = os.path.join(os.path.dirname(__file__), "..", ".cache")


def _get(url: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, url.split("://", 1)[1].replace("/", "_"))
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    with open(path, encoding="utf-8") as f:
        return f.read()


def _tsv(text: str) -> list[dict]:
    csv.field_size_limit(10**7)
    return list(csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE))


def _sample(rows: list, n: int, seed: int) -> list:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def afrimmlu(lang: str, n: int = 100, seed: int = 0) -> list[dict]:
    out = []
    for i, r in enumerate(_tsv(_get(HF.format(name="afrimmlu", lang=lang)))):
        try:
            choices = ast.literal_eval(r["choices"])
        except Exception:
            continue
        if len(choices) != 4 or r["answer"].strip() not in "ABCD":
            continue
        out.append({"id": f"mmlu-{lang}-{i}", "question": r["question"], "choices": choices,
                    "answer": r["answer"].strip(), "subject": r.get("subject", "")})
    return _sample(out, n, seed)


def afrimgsm(lang: str, n: int = 50, seed: int = 0) -> list[dict]:
    out = []
    for i, r in enumerate(_tsv(_get(HF.format(name="afrimgsm", lang=lang)))):
        try:
            ans = float(str(r["answer_number"]).replace(",", ""))
        except Exception:
            continue
        out.append({"id": f"mgsm-{lang}-{i}", "question": r["question"], "answer": ans})
    return _sample(out, n, seed)


def lafand(lang: str, n: int = 50, seed: int = 0) -> list[dict]:
    en = _get(LAFAND.format(lang=lang, side="en")).splitlines()
    xx = _get(LAFAND.format(lang=lang, side=lang)).splitlines()
    pairs = [{"id": f"mt-{lang}-{i}", "en": a.strip(), "xx": b.strip()}
             for i, (a, b) in enumerate(zip(en, xx)) if a.strip() and b.strip() and len(a) < 400]
    return _sample(pairs, n, seed)
