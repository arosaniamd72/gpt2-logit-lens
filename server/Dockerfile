FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
# Bake the GPT-2 weights into the image so a waking server doesn't have to download them.
RUN python -c "from transformers import GPT2LMHeadModel, GPT2TokenizerFast as T; T.from_pretrained('gpt2'); GPT2LMHeadModel.from_pretrained('gpt2')"
COPY --chown=user app.py .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
