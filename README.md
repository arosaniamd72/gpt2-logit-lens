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

GPT-2 doesn't run in the browser here. Generate a trace with the exporter from
the companion tooling, then paste the JSON into the page's
**Play your own prompt** box:

    python export_lens_json.py --out lens.json --prompts "your prompt here"

## How it's projected

Each layer's vector is passed through the model's own final `LayerNorm` and
unembedding matrix, turning a mid-stack state into a distribution over the
vocabulary. Layer 11's residual equals the model's real output, which is the
built-in check that the projection is done correctly.
