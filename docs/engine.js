/* GPT-2 small, from scratch, for the logit lens.
 *
 * Runs a real forward pass in the browser (or Node, for testing) so that the
 * viewer can show, per layer, both the residual stream's running prediction
 * and that layer's MLP contribution. No off-the-shelf runtime exposes the
 * per-layer MLP write, which is why this is hand-written.
 *
 * Weights are fp16 (2D) and fp32 (1D), decoded to fp32 Float32Arrays on load.
 * fp16 matches full precision to ~1e-3, which int8 did not: int8 survived
 * argmax but distorted the probability mass the lens exists to show.
 */
(function (root) {
  "use strict";

  // ---- byte-level BPE tokenizer ----------------------------------------
  function bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (bs.indexOf(b) === -1) { bs.push(b); cs.push(256 + n); n++; }
    }
    const b2u = {}, u2b = {};
    for (let i = 0; i < bs.length; i++) {
      b2u[bs[i]] = String.fromCharCode(cs[i]);
      u2b[String.fromCharCode(cs[i])] = bs[i];
    }
    return { b2u, u2b };
  }
  const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  function Tokenizer(vocab, merges) {
    const { b2u } = bytesToUnicode();
    const enc = new TextEncoder();
    const ranks = {};
    for (let i = 0; i < merges.length; i++) ranks[merges[i]] = i;
    const dec = {};
    for (const tok in vocab) dec[vocab[tok]] = tok;
    const { u2b } = bytesToUnicode();

    function bpe(word) {           // word: array of unicode-mapped chars
      if (word.length === 1) return word;
      let parts = word.slice();
      while (true) {
        let best = null, bestRank = Infinity, bestIdx = -1;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = ranks[parts[i] + " " + parts[i + 1]];
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; bestIdx = i; }
        }
        if (best === null) break;
        const merged = parts[bestIdx] + parts[bestIdx + 1];
        parts = parts.slice(0, bestIdx).concat([merged], parts.slice(bestIdx + 2));
      }
      return parts;
    }

    this.encode = function (text) {
      const ids = [];
      const pieces = text.match(PAT) || [];
      for (const piece of pieces) {
        const bytes = enc.encode(piece);
        let word = "";
        for (const byte of bytes) word += b2u[byte];
        const toks = bpe(word.split(""));
        for (const t of toks) {
          const id = vocab[t];
          if (id !== undefined) ids.push(id);
        }
      }
      return ids;
    };
    this.decode = function (ids) {
      let s = "";
      for (const id of ids) s += dec[id] || "";
      const bytes = [];
      for (const ch of s) { const b = u2b[ch]; if (b !== undefined) bytes.push(b); }
      return new TextDecoder().decode(new Uint8Array(bytes));
    };
    this.decodeOne = function (id) { return this.decode([id]); };
  }

  // ---- model -----------------------------------------------------------
  // fp16 (IEEE half) -> fp32, via a 65536-entry lookup for speed on load.
  const H2F = (function () {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
      if (e === 0) t[h] = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
      else if (e === 31) t[h] = f ? NaN : (s ? -Infinity : Infinity);
      else t[h] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return t;
  })();

  function Model(manifest, buffer) {
    this.cfg = manifest;
    const T = {}, meta = {};
    for (const t of manifest.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      let arr;
      if (t.dtype === "fp16") {
        const u16 = new Uint16Array(buffer.slice(t.offset, t.offset + n * 2));
        arr = new Float32Array(n);
        for (let i = 0; i < n; i++) arr[i] = H2F[u16[i]];
      } else {
        arr = new Float32Array(buffer.slice(t.offset, t.offset + n * 4));
      }
      T[t.name] = arr;
      meta[t.name] = t.shape;
    }
    this.T = T;
    this.shape = meta;
  }

  Model.prototype.linear = function (x, name, bias) {
    // x: Float32Array[in]; W flat Float32Array[in,out]; returns Float32Array[out]
    const W = this.T[name], sh = this.shape[name], rows = sh[0], cols = sh[1];
    const y = new Float32Array(cols);
    if (bias) y.set(this.T[bias]);
    for (let i = 0; i < rows; i++) {
      const xi = x[i]; if (xi === 0) continue;
      const base = i * cols;
      for (let j = 0; j < cols; j++) y[j] += xi * W[base + j];
    }
    return y;
  };

  Model.prototype.embRow = function (name, idx) {
    const cols = this.shape[name][1];
    return this.T[name].subarray(idx * cols, idx * cols + cols);
  };

  function layernorm(x, w, b, eps) {
    const n = x.length;
    let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; } v /= n;
    const inv = 1 / Math.sqrt(v + eps);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) y[i] = (x[i] - m) * inv * w[i] + b[i];
    return y;
  }
  function gelu(x) { // gelu_new (tanh approximation), as GPT-2 uses
    const y = new Float32Array(x.length);
    const c = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      y[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
    }
    return y;
  }

  // Full forward over a token sequence. Returns, for the LAST position only,
  // the residual stream after each block and each block's MLP output.
  Model.prototype.forwardLast = function (ids) {
    const T = this.T, cfg = this.cfg, ne = cfg.n_embd, nh = cfg.n_head, hd = ne / nh, eps = cfg.ln_eps;
    const S = ids.length;
    // hidden states for all positions (needed for attention); [S][ne]
    let x = [];
    for (let p = 0; p < S; p++) {
      const e = this.embRow("transformer.wte.weight", ids[p]);
      const pe = this.embRow("transformer.wpe.weight", p);
      const row = new Float32Array(ne);
      for (let d = 0; d < ne; d++) row[d] = e[d] + pe[d];
      x.push(row);
    }
    const residLast = [], mlpLast = [];
    for (let L = 0; L < cfg.n_layer; L++) {
      const pf = "transformer.h." + L + ".";
      const ln1w = T[pf + "ln_1.weight"], ln1b = T[pf + "ln_1.bias"];
      const ln2w = T[pf + "ln_2.weight"], ln2b = T[pf + "ln_2.bias"];

      // q,k,v for every position
      const Q = [], K = [], V = [];
      for (let p = 0; p < S; p++) {
        const h = layernorm(x[p], ln1w, ln1b, eps);
        const qkv = this.linear(h, pf + "attn.c_attn.weight", pf + "attn.c_attn.bias");
        Q.push(qkv.subarray(0, ne)); K.push(qkv.subarray(ne, 2 * ne)); V.push(qkv.subarray(2 * ne, 3 * ne));
      }
      // attention, all positions (causal)
      const attnOut = [];
      for (let p = 0; p < S; p++) {
        const out = new Float32Array(ne);
        for (let head = 0; head < nh; head++) {
          const off = head * hd;
          const scores = new Float32Array(p + 1);
          let mx = -Infinity;
          for (let t = 0; t <= p; t++) {
            let s = 0; for (let d = 0; d < hd; d++) s += Q[p][off + d] * K[t][off + d];
            s /= Math.sqrt(hd); scores[t] = s; if (s > mx) mx = s;
          }
          let sum = 0; for (let t = 0; t <= p; t++) { scores[t] = Math.exp(scores[t] - mx); sum += scores[t]; }
          for (let t = 0; t <= p; t++) { const a = scores[t] / sum; for (let d = 0; d < hd; d++) out[off + d] += a * V[t][off + d]; }
        }
        attnOut.push(this.linear(out, pf + "attn.c_proj.weight", pf + "attn.c_proj.bias"));
      }
      for (let p = 0; p < S; p++) { const r = x[p]; const a = attnOut[p]; for (let d = 0; d < ne; d++) r[d] += a[d]; }
      // MLP, all positions
      const mlpOut = [];
      for (let p = 0; p < S; p++) {
        const h = layernorm(x[p], ln2w, ln2b, eps);
        const m = gelu(this.linear(h, pf + "mlp.c_fc.weight", pf + "mlp.c_fc.bias"));
        mlpOut.push(this.linear(m, pf + "mlp.c_proj.weight", pf + "mlp.c_proj.bias"));
      }
      for (let p = 0; p < S; p++) { const r = x[p]; const m = mlpOut[p]; for (let d = 0; d < ne; d++) r[d] += m[d]; }
      residLast.push(x[S - 1].slice());
      mlpLast.push(mlpOut[S - 1]);
    }
    return { residLast, mlpLast };
  };

  // Project a d_model vector through ln_f + unembed -> full softmax probs.
  Model.prototype.lens = function (vec) {
    const T = this.T, cfg = this.cfg;
    const n = layernorm(vec, T["transformer.ln_f.weight"], T["transformer.ln_f.bias"], cfg.ln_eps);
    const W = T["transformer.wte.weight"];                 // [vocab, ne] flat
    const rows = this.shape["transformer.wte.weight"][0], cols = this.shape["transformer.wte.weight"][1];
    const logits = new Float32Array(rows);
    let mx = -Infinity;
    for (let v = 0; v < rows; v++) {
      let s = 0; const base = v * cols;
      for (let d = 0; d < cols; d++) s += n[d] * W[base + d];
      logits[v] = s; if (s > mx) mx = s;
    }
    let sum = 0;
    for (let v = 0; v < rows; v++) { const e = Math.exp(logits[v] - mx); logits[v] = e; sum += e; }
    for (let v = 0; v < rows; v++) logits[v] /= sum;
    return logits;
  };

  function topk(probs, k, decodeOne) {
    const idx = Array.from({ length: probs.length }, (_, i) => i);
    idx.sort((a, b) => probs[b] - probs[a]);
    const out = [];
    for (let i = 0; i < k; i++) out.push([decodeOne(idx[i]), Math.round(probs[idx[i]] * 1e5) / 1e5]);
    return out;
  }
  function entropy(probs) {
    let h = 0;
    for (let i = 0; i < probs.length; i++) { const p = probs[i]; if (p > 0) h -= p * Math.log(p); }
    return Math.round(h * 1e4) / 1e4;
  }
  function argmax(probs) { let bi = 0, bv = -Infinity; for (let i = 0; i < probs.length; i++) if (probs[i] > bv) { bv = probs[i]; bi = i; } return bi; }

  // Greedy generate, producing the viewer's DATA shape. onLayer/onStep are
  // optional progress callbacks.
  function generate(model, tok, prompt, newTokens, k, cb) {
    cb = cb || {};
    let ids = tok.encode(prompt);
    if (ids.length === 0) ids = [tok.encode(" ")[0] || 220];
    const nl = model.cfg.n_layer;
    const rec = { prompt, steps: [] };
    for (let step = 0; step < newTokens; step++) {
      const { residLast, mlpLast } = model.forwardLast(ids);
      const final = model.lens(residLast[nl - 1]); // == model output distribution
      const cid = argmax(final);
      const layers = [];
      for (let L = 0; L < nl; L++) {
        const rp = model.lens(residLast[L]);
        const mp = model.lens(mlpLast[L]);
        layers.push({
          resid: topk(rp, k, (i) => tok.decodeOne(i)),
          mlp: topk(mp, k, (i) => tok.decodeOne(i)),
          p_chosen: Math.round(rp[cid] * 1e5) / 1e5,
          resid_H: entropy(rp), mlp_H: entropy(mp),
        });
        if (cb.onLayer) cb.onLayer(step, L, nl);
      }
      rec.steps.push({
        context: tok.decode(ids), chosen: tok.decodeOne(cid), chosen_id: cid,
        layers, final: topk(final, k, (i) => tok.decodeOne(i)),
      });
      ids = ids.concat([cid]);
      if (cb.onStep) cb.onStep(step, newTokens);
    }
    rec.generated = tok.decode(ids);
    return { model: model.cfg.model, n_layers: nl, topk: k, prompts: [rec] };
  }

  const API = { Tokenizer, Model, generate, bytesToUnicode };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.GPT2Engine = API;
})(typeof self !== "undefined" ? self : this);
