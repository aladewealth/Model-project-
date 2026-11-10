#!/usr/bin/env python3
"""Turn results/*.jsonl into results/summary.csv, results/chart.png and a self-contained results/report.html."""
import base64
import glob
import html
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LANG = {"yor": "Yoruba", "hau": "Hausa", "ibo": "Igbo"}
TASK = {"mmlu": "AfriMMLU acc %", "mgsm": "AfriMGSM acc %",
        "mt_en2xx": "EN→X chrF++", "mt_xx2en": "X→EN chrF++"}

rows = []
for p in sorted(glob.glob("results/*.jsonl")):
    rows += [json.loads(l) for l in open(p, encoding="utf-8")]
if not rows:
    raise SystemExit("no results/*.jsonl yet - run run_eval.py first")
df = pd.DataFrame(rows)
df.loc[df.task.isin(["mmlu", "mgsm"]), "score"] *= 100

summary = df.groupby(["model", "task", "lang"]).score.agg(["mean", "count"]).reset_index()
summary["task_name"] = summary.task.map(TASK)
summary["language"] = summary.lang.map(LANG)
summary.to_csv("results/summary.csv", index=False)
pivot = summary.pivot_table(index=["task_name", "language"], columns="model", values="mean").round(1)

# Yoruba tone marks
tone = None
if "tone_density" in df:
    t = df.dropna(subset=["tone_density"])
    if len(t):
        tone = t.groupby("model")[["tone_density", "tone_density_ref"]].mean()
        tone["fidelity %"] = (100 * tone.tone_density / tone.tone_density_ref).round(1)

# chart
models = sorted(df.model.unique())
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (tk, label) in zip(axes, TASK.items()):
    sub = summary[summary.task == tk]
    langs = [l for l in LANG if l in set(sub.lang)]
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [sub[(sub.model == m) & (sub.lang == l)]["mean"].mean() for l in langs]
        ax.bar([j + i * w for j in range(len(langs))], vals, w, label=m)
    ax.set_xticks([j + w * (len(models) - 1) / 2 for j in range(len(langs))])
    ax.set_xticklabels([LANG[l] for l in langs])
    ax.set_title(label)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/chart.png", dpi=130)
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110)
img = base64.b64encode(buf.getvalue()).decode()

# examples: Yoruba translations where models disagree most
ex_html = ""
mt = df[(df.task == "mt_en2xx") & (df.lang == "yor")]
if len(models) >= 2 and len(mt):
    wide = mt.pivot_table(index="id", columns="model", values="score")
    wide["gap"] = (wide[models[0]] - wide[models[-1]]).abs()
    for iid in wide.sort_values("gap", ascending=False).index[:5]:
        r = mt[mt.id == iid]
        src = html.escape(r.iloc[0]["input"]); ref = html.escape(r.iloc[0]["gold"])
        outs = "".join(f"<li><b>{html.escape(m)}</b> ({r[r.model==m].score.iloc[0]:.0f}): "
                       f"{html.escape(r[r.model==m].output.iloc[0])}</li>" for m in models)
        ex_html += f"<div class=ex><p><i>EN:</i> {src}</p><p><i>Ref:</i> {ref}</p><ul>{outs}</ul></div>"

tone_html = tone[["fidelity %"]].to_html() if tone is not None else "<p>n/a</p>"
page = f"""<!doctype html><meta charset=utf-8><title>Naija Eval</title>
<style>body{{font:15px/1.5 system-ui;max-width:1000px;margin:40px auto;padding:0 16px;color:#111}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:4px 10px;text-align:right}}
.ex{{border-left:3px solid #0a7;padding:4px 12px;margin:12px 0;background:#f7faf9}}img{{max-width:100%}}</style>
<h1>Naija Eval</h1>
<p>How much does N-ATLaS's Nigerian-language fine-tune actually buy over its Llama-3 8B base? Same prompts, greedy decoding,
4-bit on a free Kaggle T4. Benchmarks: AfriMMLU, AfriMGSM (Masakhane), LAFAND-MT news translation.</p>
<img src="data:image/png;base64,{img}">
<h2>Scores</h2>{pivot.to_html()}
<h2>Yoruba tone-mark fidelity</h2><p>Tone marks and under-dots per letter in EN→Yoruba output, as % of the human reference. 100% means the model writes Yoruba as densely marked as a native translator.</p>{tone_html}
<h2>Where they differ most (EN→Yoruba)</h2>{ex_html or '<p>n/a</p>'}
<p style="color:#666;font-size:13px">N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.</p>"""
open("results/report.html", "w", encoding="utf-8").write(page)
print(pivot.to_string()); print(tone if tone is not None else "")
print("wrote results/summary.csv, results/chart.png, results/report.html")
