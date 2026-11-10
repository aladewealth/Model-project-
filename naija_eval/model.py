"""Thin wrapper: load a chat model (4-bit on GPU, fp32 on CPU) and run batched chat generation."""
from __future__ import annotations

from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are a helpful assistant fluent in English, Yoruba, Hausa and Igbo."


class ChatModel:
    def __init__(self, name: str, four_bit: bool = True, max_batch: int = 8):
        self.name = name
        self.max_batch = max_batch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {}
        if torch.cuda.is_available():
            kw["device_map"] = "auto"
            if four_bit:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16)
            else:
                kw["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
        self.device = next(self.model.parameters()).device
        # Llama-3 style models end turns with <|eot_id|>
        eos = [self.tok.eos_token_id]
        eot = self.tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot != self.tok.unk_token_id:
            eos.append(eot)
        self.eos = list({e for e in eos if e is not None})

    def prompt(self, user: str, system: str = SYSTEM) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                date_string=datetime.now().strftime("%d %b %Y"))
        except Exception:
            return f"{system}\n\nUser: {user}\nAssistant:"

    @torch.no_grad()
    def generate(self, users: list[str], max_new_tokens: int = 256) -> list[str]:
        outs: list[str] = []
        for i in range(0, len(users), self.max_batch):
            texts = [self.prompt(u) for u in users[i:i + self.max_batch]]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=1.1, eos_token_id=self.eos,
                                      pad_token_id=self.tok.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                outs.append(self.tok.decode(row, skip_special_tokens=True).strip())
        return outs

    @torch.no_grad()
    def choice_logprobs(self, users: list[str], options=("A", "B", "C", "D")) -> list[list[float]]:
        """Score multiple choice by the next-token logit of each letter after the assistant header."""
        ids = [self.tok.encode(o, add_special_tokens=False)[0] for o in options]
        res = []
        for i in range(0, len(users), self.max_batch):
            texts = [self.prompt(u) for u in users[i:i + self.max_batch]]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            logits = self.model(**enc).logits[:, -1, :].float()
            res.extend(torch.log_softmax(logits[:, ids], dim=-1).tolist())
        return res
