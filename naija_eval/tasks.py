"""Task prompts and scoring. Every task returns per-item records so the report can show real examples."""
from __future__ import annotations

import re
import unicodedata

from . import data
from .data import LANG_NAMES

# ---------- Yoruba tone-mark fidelity ----------
# Yoruba is tonal; written correctly it carries acute/grave accents and under-dots (ẹ ọ ṣ).
# N-ATLaS's own card rates Yoruba fluency 2.71/5, so we measure whether outputs keep these marks.
_MARKS = {"\u0300", "\u0301", "\u0323"}  # grave, acute, dot below


def tone_mark_density(text: str) -> float:
    letters = [c for c in unicodedata.normalize("NFD", text) if c.isalpha()]
    if not letters:
        return 0.0
    marks = sum(1 for c in unicodedata.normalize("NFD", text) if c in _MARKS)
    return marks / len(letters)


# ---------- AfriMMLU ----------
def run_mmlu(model, lang: str, n: int, seed: int) -> list[dict]:
    items = data.afrimmlu(lang, n, seed)
    prompts = [
        f"{it['question']}\n\nA. {it['choices'][0]}\nB. {it['choices'][1]}\n"
        f"C. {it['choices'][2]}\nD. {it['choices'][3]}\n\nAnswer with only the letter A, B, C or D."
        for it in items
    ]
    scores = model.choice_logprobs(prompts)
    out = []
    for it, s in zip(items, scores):
        pred = "ABCD"[max(range(4), key=lambda k: s[k])]
        out.append({"task": "mmlu", "lang": lang, "id": it["id"], "pred": pred, "gold": it["answer"],
                    "score": float(pred == it["answer"]), "input": it["question"][:300]})
    return out


# ---------- AfriMGSM ----------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str):
    nums = _NUM.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def run_mgsm(model, lang: str, n: int, seed: int) -> list[dict]:
    items = data.afrimgsm(lang, n, seed)
    prompts = [f"{it['question']}\n\nSolve step by step, then give the final answer as a number on the last line."
               for it in items]
    gens = model.generate(prompts, max_new_tokens=320)
    out = []
    for it, g in zip(items, gens):
        pred = _last_number(g)
        ok = pred is not None and abs(pred - it["answer"]) < 1e-6
        out.append({"task": "mgsm", "lang": lang, "id": it["id"], "pred": pred, "gold": it["answer"],
                    "score": float(ok), "input": it["question"][:300], "output": g[-600:]})
    return out


# ---------- Translation (LAFAND-MT) ----------
def run_mt(model, lang: str, n: int, seed: int) -> list[dict]:
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)  # chrF++
    items = data.lafand(lang, n, seed)
    name = LANG_NAMES[lang]
    out = []
    for direction in ("en2xx", "xx2en"):
        if direction == "en2xx":
            prompts = [f"Translate this English sentence into {name}. Reply with the translation only.\n\n{it['en']}"
                       for it in items]
            refs = [it["xx"] for it in items]
            srcs = [it["en"] for it in items]
        else:
            prompts = [f"Translate this {name} sentence into English. Reply with the translation only.\n\n{it['xx']}"
                       for it in items]
            refs = [it["en"] for it in items]
            srcs = [it["xx"] for it in items]
        gens = model.generate(prompts, max_new_tokens=200)
        for it, src, g, ref in zip(items, srcs, gens, refs):
            hyp = g.strip().split("\n")[0].strip().strip('"')
            rec = {"task": f"mt_{direction}", "lang": lang, "id": it["id"], "input": src, "output": hyp,
                   "gold": ref, "score": chrf.sentence_score(hyp, [ref]).score}
            if lang == "yor" and direction == "en2xx":
                rec["tone_density"] = tone_mark_density(hyp)
                rec["tone_density_ref"] = tone_mark_density(ref)
            out.append(rec)
    return out


TASKS = {"mmlu": run_mmlu, "mgsm": run_mgsm, "mt": run_mt}
