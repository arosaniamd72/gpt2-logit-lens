# GPT-2 Logit Lens

An interactive, slow-motion view of a GPT-2 prediction forming layer by layer.

Pick a prompt and press **Replay in slow motion**. A scan descends the 12-layer
stack. Each layer makes two writes to the residual stream, attention then MLP,
and the page shows them two ways:

- **Ledger.** Pick a token (and optionally a rival). Its score starts at the
  *prior*, the final LayerNorm's bias read through the unembedding, which leans
  toward frequent tokens and is the same for every prompt. Each write then adds
  a *vote*, and a *shrink*: the final LayerNorm divides the stream by its size,
  so a write that enlarges the stream turns down every earlier vote and pulls
  the score back toward the prior. Both are exact; the running total equals the
  logit-lens reading after every write, and the last row is the real output.
- **Before → after.** For each write, the lens reading of the stream just
  before and just after it (the tokens that moved most) and how much the write
  multiplied the stream's size, then the stream's reading after the layer.

Side charts track p(emitted token), the stream's size (σ) and entropy, layer by
layer. Six precomputed examples are baked into the page and work offline.

## Play your own prompt

Type a prompt and press **Run**. The page asks the GPT-2 server (`server/`, hosted as the
Hugging Face Space `arosaniamd/gpt2-lens` at https://gpt2.clinicallinks.net) for the
layer-by-layer trace. If the server can't be reached, it falls back to running GPT-2 in the
browser (`docs/engine.js` + `docs/worker.js`, one-time ~248 MB download). Both produce the
same JSON shape.

## Ask Claude

Below the viewer, readers can ask questions about LLMs and what the viewer shows. The page
sends the question plus the on-screen layer readings to the server's `/ask` endpoint, which
calls Claude through the Anthropic API (key in the Space secret `ANTHROPIC_API_KEY`; model
set by `CLAUDE_MODEL`, default `claude-sonnet-5-5`). Off-topic questions are declined.
Limits: 8 questions per visitor per 10 minutes and 150 per day (`ASK_PER_IP_PER_10MIN`,
`ASK_PER_DAY` Space variables).

Pushing changes under `server/` redeploys the Space (`.github/workflows/hf-space.yml`).
The viewer itself is published at https://tools.clinicallinks.net/logit-lens/ by the
`arosaniamd72/clinical-tools` repository.

## How it's projected

Each layer's vector is passed through the model's own final `LayerNorm` and
unembedding matrix, turning a mid-stack state into a distribution over the
vocabulary. Layer 11's residual equals the model's real output, which is the
built-in check that the projection is done correctly.

For the ledger, token *j*'s score (its logit minus the vocabulary-average logit)
is `content + prior`, with `content = (x − mean)/σ · γ(W_U[j] − mean W_U)` and
`prior = β · (W_U[j] − mean W_U)`. When a write *c* lands,
`content_after = content_before · σ_before/σ_after + vote`, where
`vote = (c − mean)/σ_after · γ(W_U[j] − mean W_U)`. The change in the first
term is the shrink.
