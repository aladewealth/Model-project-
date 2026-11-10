# Naija Eval

**What did Nigeria's national LLM fine-tune actually buy?**

[N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) is a free, open Llama-3 8B fine-tuned on ~392M tokens of Yoruba, Hausa, Igbo and English instructions. Its own model card reports human ratings of 3.98/5 for Hausa, 3.87/5 for Igbo and only **2.69/5 for Yoruba**, but nothing against public benchmarks or against the model it started from.

Naija Eval runs N-ATLaS and its exact base model (Llama-3 8B Instruct) on the same prompts, so every point of difference is the fine-tune. It runs on a **free Kaggle T4** in 4-bit, with no paid compute.

## What it measures

| Task | Source | Metric | Default n per language |
|---|---|---|---|
| Knowledge (multiple choice) | AfriMMLU (Masakhane) | accuracy, letter log-probs | 100 |
| Math reasoning | AfriMGSM (Masakhane) | exact-match final number | 40 |
| Translation EN→X and X→EN | LAFAND-MT news (Masakhane) | chrF++ | 50 each way |
| **Yoruba tone-mark fidelity** | LAFAND-MT EN→Yoruba | tone marks per letter vs human reference | 50 |

Languages: Yoruba (`yor`), Hausa (`hau`), Igbo (`ibo`). Fixed seed, greedy decoding.

The tone-mark metric is the new bit. Yoruba is tonal: correctly written, it carries acute and grave accents and under-dots (ẹ, ọ, ṣ). A model that drops them writes text that is ambiguous to readers and breaks text-to-speech. This tests the model card's low Yoruba fluency score directly.

## Run it

```bash
pip install -r requirements.txt
huggingface-cli login   # after accepting the N-ATLaS terms on its model page
python run_eval.py --model NCAIR1/N-ATLaS --tag N-ATLaS
python run_eval.py --model unsloth/llama-3-8b-Instruct --tag Llama-3-8B-Instruct
python report.py        # results/report.html, chart.png, summary.csv
```

On Kaggle: open `notebooks/naija_eval_kaggle.ipynb`, set GPU T4 and Internet on, add an `HF_TOKEN` secret, then run all. Any Hugging Face chat model works with `--model`, so you can add Aya, Qwen or a fine-tune of your own.

## Layout

```
naija_eval/data.py    seeded slices of AfriMMLU, AfriMGSM, LAFAND-MT (downloaded and cached)
naija_eval/model.py   4-bit loading, batched chat generation, multiple-choice scoring
naija_eval/tasks.py   prompts, scoring, Yoruba tone-mark density
run_eval.py           one model -> results/<tag>.jsonl (resumable)
report.py             all models -> summary table, chart, HTML report with examples
```

## Limitations

Small samples (a few hundred items per model), so treat gaps under ~3 points as noise. chrF++ rewards surface overlap, not meaning. AfriMMLU and AfriMGSM are translated from English benchmarks, so they test cross-lingual transfer more than Nigerian cultural knowledge.

## Attribution

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies. Benchmarks by the Masakhane community. Built by Alade Wealth (Atoms Intelligence Labs) for the Geek Room Global AI Hackathon 2026.
