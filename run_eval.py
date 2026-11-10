#!/usr/bin/env python3
"""Run Naija Eval for one model.

  python run_eval.py --model NCAIR1/N-ATLaS --tag n-atlas
  python run_eval.py --model unsloth/llama-3-8b-Instruct --tag llama3-8b-base

Writes results/<tag>.jsonl (one record per item). Re-running skips tasks already done.
"""
import argparse
import json
import os
import time

from naija_eval.model import ChatModel
from naija_eval.tasks import TASKS

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True, help="short name used in the report")
ap.add_argument("--langs", default="yor,hau,ibo")
ap.add_argument("--tasks", default="mmlu,mgsm,mt")
ap.add_argument("--n-mmlu", type=int, default=100)
ap.add_argument("--n-mgsm", type=int, default=40)
ap.add_argument("--n-mt", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--no-4bit", action="store_true")
args = ap.parse_args()

os.makedirs("results", exist_ok=True)
path = f"results/{args.tag}.jsonl"
done = set()
if os.path.exists(path):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        done.add((r["task"].split("_")[0], r["lang"]))

model = ChatModel(args.model, four_bit=not args.no_4bit, max_batch=args.batch)
sizes = {"mmlu": args.n_mmlu, "mgsm": args.n_mgsm, "mt": args.n_mt}

with open(path, "a", encoding="utf-8") as f:
    for task in args.tasks.split(","):
        for lang in args.langs.split(","):
            if (task, lang) in done:
                print(f"skip {task}/{lang} (already in {path})")
                continue
            t0 = time.time()
            recs = TASKS[task](model, lang, sizes[task], args.seed)
            for r in recs:
                r["model"] = args.tag
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            avg = sum(r["score"] for r in recs) / max(len(recs), 1)
            print(f"{args.tag} {task}/{lang}: n={len(recs)} avg={avg:.3f} ({time.time()-t0:.0f}s)")
print("wrote", path)
