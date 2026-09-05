/* Loads the GPT-2 weights once, then runs generation off the main thread so
 * the UI stays responsive. Reports download and compute progress. */
importScripts("engine.js");

let model = null, tok = null;

async function load(post) {
  const base = new URL("model/", location.href).href;
  const man = await (await fetch(base + "manifest.json")).json();
  const tokj = await (await fetch(base + "tokenizer.json")).json();

  const total = man.total_bytes;
  let got = 0;
  const shardBufs = [];
  for (const s of man.shards) {
    const resp = await fetch(base + s);
    const reader = resp.body.getReader();
    const chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      post({ type: "progress", phase: "download", pct: got / total });
    }
    let len = 0; for (const c of chunks) len += c.length;
    const b = new Uint8Array(len); let o = 0;
    for (const c of chunks) { b.set(c, o); o += c.length; }
    shardBufs.push(b);
  }
  let tl = 0; for (const b of shardBufs) tl += b.length;
  let all = new Uint8Array(tl); let o = 0;
  for (const b of shardBufs) { all.set(b, o); o += b.length; }
  post({ type: "progress", phase: "init", pct: 1 });

  model = new GPT2Engine.Model(man, all.buffer);
  tok = new GPT2Engine.Tokenizer(tokj.vocab, tokj.merges);
  all = null; shardBufs.length = 0; // free the fp16 source; weights are now fp32
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (!model) {
      self.postMessage({ type: "progress", phase: "download", pct: 0 });
      await load(self.postMessage.bind(self));
    }
    if (m.type === "gen") {
      const data = GPT2Engine.generate(model, tok, m.prompt, m.newTokens, m.topk, {
        onStep: (s, n) => self.postMessage({ type: "progress", phase: "compute", pct: (s + 1) / n }),
      });
      self.postMessage({ type: "done", data });
    }
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
