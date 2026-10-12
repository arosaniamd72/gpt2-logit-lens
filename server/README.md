---
title: GPT-2 Logit Lens
emoji: 🔭
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: GPT-2 small logit-lens API for the Logit Lens viewer
---

Backend for the GPT-2 Logit Lens viewer at https://tools.clinicallinks.net/logit-lens/.
`POST /lens {"prompt": "...", "new_tokens": 5, "topk": 8}` returns, for each generated
token, what every layer's residual stream, attention write and MLP write predict.
Source: https://github.com/arosaniamd72/gpt2-logit-lens (server/).
