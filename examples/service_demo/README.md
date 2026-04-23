# DaseR Service Demo

End-to-end walkthrough of the DaseR service layer: upload two small
documents, list them, run inference over both of them, then delete one.

The demo drives the public HTTP API only. `vllm serve` and
`python -m daser.service` must be running first.

## 1. Install service extras

```bash
pip install -e '.[service]'
```

## 2. Start vLLM

```bash
vllm serve <model-path> \
  --port 8001 \
  --kv-transfer-config '{
    "kv_connector": "DaserConnector",
    "kv_connector_module_path": "daser.connector.daser_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "socket_path": "/tmp/daser.sock",
      "store_path": "/tmp/daser_demo/daser.store",
      "block_tokens": 16,
      "model_id": "demo"
    }
  }'
```

## 3. Start the DaseR service (embedded control plane + HTTP API)

```bash
python -m daser.service \
  --host 0.0.0.0 --port 8080 \
  --vllm-base-url http://127.0.0.1:8001 \
  --model <model-path> \
  --tokenizer <model-path> \
  --socket-path /tmp/daser.sock \
  --store-path /tmp/daser_demo/daser.store \
  --index-path /tmp/daser_demo/daser.index \
  --block-tokens 16 \
  --chunk-blocks 4
```

> `--model` / `--tokenizer` should be the same path you passed to
> `vllm serve`. The tokenizer is loaded inside the service process to
> keep chunk keys consistent with what vLLM sees.

## 4. Run the demo

```bash
python examples/service_demo/demo.py --service-url http://127.0.0.1:8080
```

Expected output (truncated):

```
==> health
{ "status": "ok", "vllm": true }

==> upload doc A
{ "doc_id": "...", "status": "ready", "chunk_count": 1, "chunk_count_cached": 1, "prefill_ms": 180.2 }
...
==> infer over both docs
{ "text": "...", "prompt_tokens": 513, "completion_tokens": 128, "latency_ms": 612.8 }
```

The second upload of the same document is a no-op: the chunk keys hash
to the same values, so `register_doc` just attaches a new `doc_id` to
the existing `ChunkMeta.doc_ids` list.

## Troubleshooting

- **Connection refused to the socket**: make sure the paths passed to
  `vllm serve` and `daser.service` agree on `socket_path`.
- **`doc N has no cached tokens for prompt rebuild`**: a doc must be
  uploaded through `/documents` before `/infer` can use it; inferring
  against a doc that was evicted requires re-uploading.
- **vLLM rejects `max_tokens=0`**: the service uses `max_tokens=1` for
  prefill and discards the single decoded token.
