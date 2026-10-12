"""GPT-2 logit-lens server for the GPT-2 Logit Lens viewer.

POST /lens {"prompt": str, "new_tokens": int, "topk": int}
returns the same JSON shape the in-browser engine (docs/engine.js -> generate) produces,
so the page can use either interchangeably.
"""
import math
import threading

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)
torch.set_num_threads(2)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
tr = model.transformer
N_LAYER = model.config.n_layer
LOCK = threading.Lock()

app = FastAPI(title="GPT-2 logit lens")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

# Capture, for each block, its attention output, MLP output and resulting residual stream.
_cap = {}
for L, block in enumerate(tr.h):
    block.attn.register_forward_hook(lambda m, i, o, L=L: _cap.__setitem__(("attn", L), o[0]))
    block.mlp.register_forward_hook(lambda m, i, o, L=L: _cap.__setitem__(("mlp", L), o))
    block.register_forward_hook(lambda m, i, o, L=L: _cap.__setitem__(("resid", L), o[0] if isinstance(o, tuple) else o))


def decode_one(i):
    return tok.decode([i], clean_up_tokenization_spaces=False)


def topk(probs, k):
    v, idx = torch.topk(probs, k)
    return [[decode_one(int(i)), round(float(p), 5)] for p, i in zip(v, idx)]


def entropy(probs):
    p = probs[probs > 0]
    return round(float(-(p * p.log()).sum()), 4)


def lens(vecs):
    """Project [n, d] vectors through ln_f + unembedding -> [n, vocab] probabilities."""
    return torch.softmax(model.lm_head(tr.ln_f(vecs)).float(), dim=-1)


def generate(prompt, new_tokens, k):
    ids = tok.encode(prompt) or [220]
    rec = {"prompt": prompt, "steps": []}
    for _ in range(new_tokens):
        _cap.clear()
        model(torch.tensor([ids]))
        resid = torch.stack([_cap[("resid", L)][0, -1] for L in range(N_LAYER)])
        attn = torch.stack([_cap[("attn", L)][0, -1] for L in range(N_LAYER)])
        mlp = torch.stack([_cap[("mlp", L)][0, -1] for L in range(N_LAYER)])
        rp, ap, mp = lens(resid), lens(attn), lens(mlp)
        final = rp[N_LAYER - 1]  # == the model's real output distribution
        cid = int(torch.argmax(final))
        layers = [{
            "resid": topk(rp[L], k), "attn": topk(ap[L], k), "mlp": topk(mp[L], k),
            "p_chosen": round(float(rp[L, cid]), 5),
            "resid_H": entropy(rp[L]), "attn_H": entropy(ap[L]), "mlp_H": entropy(mp[L]),
        } for L in range(N_LAYER)]
        rec["steps"].append({
            "context": tok.decode(ids, clean_up_tokenization_spaces=False),
            "chosen": decode_one(cid), "chosen_id": cid,
            "layers": layers, "final": topk(final, k),
        })
        ids = ids + [cid]
    rec["generated"] = tok.decode(ids, clean_up_tokenization_spaces=False)
    return {"model": "gpt2", "n_layers": N_LAYER, "topk": k, "prompts": [rec]}


class LensRequest(BaseModel):
    prompt: str
    new_tokens: int = 5
    topk: int = 8


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "model": "gpt2"}


@app.post("/lens")
def run_lens(req: LensRequest):
    prompt = req.prompt.strip()
    if not prompt or len(prompt) > 400:
        raise HTTPException(400, "prompt must be 1-400 characters")
    if len(tok.encode(prompt)) > 128:
        raise HTTPException(400, "prompt is too long (128 tokens max)")
    n = max(1, min(int(req.new_tokens), 20))
    k = max(1, min(int(req.topk), 20))
    with LOCK:
        return generate(prompt, n, k)
