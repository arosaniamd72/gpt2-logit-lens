# GPT-2 Logit Lens

An interactive, slow-motion view of a GPT-2 prediction forming layer by layer.

Pick a prompt and press **Replay in slow motion**. A scan descends the 12-layer
stack, revealing at each layer what the residual stream currently predicts and
what that layer's MLP contributed on its own. Two side charts track how much the
model backs the token it eventually emits, and how committed it is (entropy),
layer by layer.

Everything runs from real GPT-2 small activations baked into the page — no
server, no model download, works offline.

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
