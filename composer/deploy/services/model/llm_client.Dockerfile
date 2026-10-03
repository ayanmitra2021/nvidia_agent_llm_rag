# --- services/llm-client/Dockerfile ---
FROM python:3.11-slim

WORKDIR /app

COPY llm_client-requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r llm_client-requirements.txt fastembed==0.8.0 && pip check

RUN mkdir -p /opt/models/fastembed && \
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 300 \
      https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz \
      --output /tmp/all-MiniLM-L6-v2.tar.gz && \
    echo "2735afe656e156af64ed603dbb1c96f3cae7f937286a8feb27fff7fa979f6a77  /tmp/all-MiniLM-L6-v2.tar.gz" | sha256sum --check --strict && \
    tar -xzf /tmp/all-MiniLM-L6-v2.tar.gz -C /opt/models/fastembed && \
    rm /tmp/all-MiniLM-L6-v2.tar.gz

RUN curl --fail --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 60 \
      https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/826711e54e001c83835913827a843d8dd0a1def9/LICENSE \
      --output /opt/models/fastembed/fast-all-MiniLM-L6-v2/LICENSE && \
    echo "1e66d43b04a3f2428303ad3316d1fbb996991541192892b22d21f0065a093b2b  /opt/models/fastembed/fast-all-MiniLM-L6-v2/LICENSE" | sha256sum --check --strict

RUN mkdir -p /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/onnx && \
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 300 \
      https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/resolve/233902d25c440f23af6f7d6e94d2946bac0bee0a/onnx/model.onnx \
      --output /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/onnx/model.onnx && \
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 60 \
      https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/raw/233902d25c440f23af6f7d6e94d2946bac0bee0a/config.json \
      --output /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/config.json && \
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 60 \
      https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/raw/233902d25c440f23af6f7d6e94d2946bac0bee0a/tokenizer.json \
      --output /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/tokenizer.json && \
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 60 \
      https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/raw/233902d25c440f23af6f7d6e94d2946bac0bee0a/tokenizer_config.json \
      --output /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/tokenizer_config.json && \
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 60 \
      https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/raw/233902d25c440f23af6f7d6e94d2946bac0bee0a/special_tokens_map.json \
      --output /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/special_tokens_map.json && \
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 60 \
      https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/raw/233902d25c440f23af6f7d6e94d2946bac0bee0a/vocab.txt \
      --output /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/vocab.txt && \
    curl --fail --location --retry 5 --retry-all-errors --connect-timeout 15 --max-time 60 \
      https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/raw/233902d25c440f23af6f7d6e94d2946bac0bee0a/README.md \
      --output /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/MODEL_CARD.md && \
    echo "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a  /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/onnx/model.onnx" | sha256sum --check --strict && \
    echo "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc  /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/config.json" | sha256sum --check --strict && \
    echo "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66  /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/tokenizer.json" | sha256sum --check --strict && \
    echo "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8  /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/tokenizer_config.json" | sha256sum --check --strict && \
    echo "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6  /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/special_tokens_map.json" | sha256sum --check --strict && \
    echo "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3  /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/vocab.txt" | sha256sum --check --strict && \
    echo "7c0ec39941d0d1d766ccde671f592971d82fe45b7eea550a80d1ea864ebc3baa  /opt/models/fastembed/cross-encoder-ms-marco-MiniLM-L6-v2/MODEL_CARD.md" | sha256sum --check --strict

ENV HF_HUB_OFFLINE=1

# Copy application
COPY llm_client.py local_retrieval.py llm_output.py llm_request.py llm_tracing.py proxy_status.html ./

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

ENTRYPOINT ["python", "llm_client.py"]
