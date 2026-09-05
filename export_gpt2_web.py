"""Export GPT-2 for in-browser inference: int8 weights + tokenizer + a
reference trace to verify a JS re-implementation against.

    python export_gpt2_web.py --model gpt2 --out web_model

Produces, under --out:
  manifest.json        tensor table, config, shard list
  weights_000.bin ...  fp16 weight data (2D) / fp32 (1D), sharded <45MB
  tokenizer.json       {vocab: token->id, merges: [...]}
  reference.json       per-prompt greedy trace (token ids + per-layer top-k),
                       computed with the SAME lens convention the browser uses
                       (ln_f applied uniformly to every layer's residual)

Weight layout in the logical byte stream, per tensor at its manifest offset:
  fp16 2D:  uint16[prod(shape)]   (IEEE half; decoded to fp32 in the browser)
  fp32 1D:  float32[prod(shape)]  (LayerNorm + biases stay full precision)
JS fetches every shard, concatenates to one ArrayBuffer, slices by offset.
"""
import argparse, json, os, math
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEFAULT_PROMPTS = [
    "The capital of France is",
    "The opposite of hot is",
    "Barack Obama was born in",
]
SHARD = 45 * 1024 * 1024

p = argparse.ArgumentParser()
p.add_argument("--model", default="gpt2")
p.add_argument("--out", default="web_model")
p.add_argument("--new-tokens", type=int, default=5)
p.add_argument("--topk", type=int, default=8)
p.add_argument("--prompts", nargs="*", default=None)
p.add_argument("--skip-tokenizer", action="store_true")  # for offline smoke tests
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
model = GPT2LMHeadModel.from_pretrained(args.model).eval()
cfg = model.config
sd = model.state_dict()

# ---- pack weights -------------------------------------------------------
buf = bytearray()
tensors = []
KEEP_FP32_SUFFIX = ("ln_1.weight","ln_1.bias","ln_2.weight","ln_2.bias",
                    "ln_f.weight","ln_f.bias",".bias")  # 1D stays fp32

def want(name):
    if name.startswith("transformer.h.") or name in (
        "transformer.wte.weight","transformer.wpe.weight",
        "transformer.ln_f.weight","transformer.ln_f.bias"):
        return not name.endswith(".attn.bias") and not name.endswith(".attn.masked_bias")
    return False

for name, t in sd.items():
    if not want(name):
        continue
    arr = t.detach().cpu().numpy().astype(np.float32)
    half = arr.ndim == 2                       # 2D weights -> fp16; 1D -> fp32
    off = len(buf)
    if half:
        buf += arr.astype(np.float16).tobytes()
    else:
        buf += arr.astype(np.float32).tobytes()
    tensors.append({"name": name, "shape": list(arr.shape),
                    "dtype": "fp16" if half else "fp32",
                    "offset": off, "length": len(buf) - off})

# ---- shard --------------------------------------------------------------
shards = []
for i in range(0, len(buf), SHARD):
    fn = f"weights_{len(shards):03d}.bin"
    with open(os.path.join(args.out, fn), "wb") as f:
        f.write(buf[i:i+SHARD])
    shards.append(fn)

manifest = {
    "model": args.model,
    "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
    "vocab_size": cfg.vocab_size, "n_positions": cfg.n_positions,
    "ln_eps": cfg.layer_norm_epsilon,
    "total_bytes": len(buf), "shard_size": SHARD,
    "shards": shards, "tensors": tensors,
}
with open(os.path.join(args.out, "manifest.json"), "w") as f:
    json.dump(manifest, f)
print(f"weights: {len(buf)/1e6:.1f} MB across {len(shards)} shard(s), {len(tensors)} tensors")

# ---- tokenizer ----------------------------------------------------------
tok = None
if not args.skip_tokenizer:
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    vocab = tok.get_vocab()  # token(str) -> id
    merges = []
    mf = tok.save_vocabulary(args.out)  # writes vocab.json + merges.txt
    merges_path = [x for x in mf if x.endswith("merges.txt")][0]
    with open(merges_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                merges.append(line)
    with open(os.path.join(args.out, "tokenizer.json"), "w") as f:
        json.dump({"vocab": vocab, "merges": merges}, f)
    for x in mf:  # keep the dir to just tokenizer.json
        try: os.remove(x)
        except OSError: pass
    print(f"tokenizer: {len(vocab)} tokens, {len(merges)} merges")

# ---- reference trace (uniform-ln_f lens; matches the JS engine) ---------
mlp_out = {}
def mk(i):
    def h(_m,_i,o): mlp_out[i] = o.detach()
    return h
for i, blk in enumerate(model.transformer.h):
    blk.mlp.register_forward_hook(mk(i))
ln_f, W_U = model.transformer.ln_f, model.lm_head

def lens(v): return torch.softmax(W_U(ln_f(v)), dim=-1)
def dec(i): return tok.decode([i]) if tok is not None else f"<{i}>"
def topk(pr):
    v, idx = pr.topk(args.topk)
    return [[dec(int(i)), round(float(x),5)] for x,i in zip(v.tolist(), idx.tolist())]
def H(pr):
    q = pr[pr>0]; return round(float(-(q*q.log()).sum()),4)

ref = {"model": args.model, "n_layers": cfg.n_layer, "topk": args.topk, "prompts": []}
with torch.no_grad():
    for prompt in (args.prompts or DEFAULT_PROMPTS):
        if tok is not None:
            ids = tok(prompt, return_tensors="pt").input_ids
        else:
            ids = torch.arange(10, 10+max(1,len(prompt.split()))).unsqueeze(0)
        rec = {"prompt": prompt, "steps": []}
        for _ in range(args.new_tokens):
            o = model(ids, output_hidden_states=True)
            pos = -1
            final = torch.softmax(o.logits[0,pos], dim=-1)
            cid = int(final.argmax())
            hs = o.hidden_states  # [emb, blk0, ..., blk11] pre-ln_f
            layers = []
            for L in range(cfg.n_layer):
                rp = lens(hs[L+1][0,pos]); mp = lens(mlp_out[L][0,pos])
                layers.append({"resid": topk(rp), "mlp": topk(mp),
                               "p_chosen": round(float(rp[cid]),5),
                               "resid_H": H(rp), "mlp_H": H(mp)})
            rec["steps"].append({
                "context_ids": ids[0].tolist(),
                "context": tok.decode(ids[0]) if tok is not None else " ".join(f"<{i}>" for i in ids[0].tolist()),
                "chosen": dec(cid), "chosen_id": cid,
                "layers": layers, "final": topk(final)})
            ids = torch.cat([ids, torch.tensor([[cid]])], dim=1)
        rec["generated"] = tok.decode(ids[0]) if tok is not None else None
        ref["prompts"].append(rec)
        print("ref:", repr(prompt))
with open(os.path.join(args.out, "reference.json"), "w") as f:
    json.dump(ref, f, separators=(",",":"))
print("wrote", args.out)
