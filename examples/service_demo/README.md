# DaseR Server RAG Demo

End-to-end walkthrough of the DaseR HTTP API: upload two small documents, list
them, run inference over both of them, then delete one.

The demo drives the public HTTP API only. Start `python -m daser.server` first,
after `vllm serve` is already listening with the DaseR connector enabled.

## 1. Install DaseR

```bash
pip install -e .
```

## 2. Start vLLM

```bash
vllm serve <model-path> \
  --port 8001 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"DaserConnector","kv_connector_module_path":"daser.connector.daser_connector","kv_role":"kv_both","kv_connector_extra_config":{"socket_path":"/tmp/daser.sock"}}'
```

`--no-enable-prefix-caching` disables vLLM's in-GPU prefix cache so the demo's
warm-run cache hits come from DaseR. Without it, vLLM may satisfy part of the
prompt from its own prefix cache and hide whether DaseR loaded KV from NVMe.

## 3. Start DaseR server

```bash
python -m daser.server \
  --host 0.0.0.0 --port 2026 \
  --vllm-base-url http://127.0.0.1:8001 \
  --store-dir /tmp/daser_demo \
  --l2-size 10gb \
  --socket-path /tmp/daser.sock
```

DaseR reads vLLM's served model id from `/v1/models`, derives KV geometry from
`<model-path>/config.json`, and creates `/tmp/daser_demo/daser.store`. If vLLM
serves a model alias instead of a local path, add `--model-path <model-path>` to
the DaseR command.

## 4. Run the demo

```bash
python examples/service_demo/demo.py --service-url http://127.0.0.1:2026
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
to the same values, so `ServerCore.register_document` just attaches a
new `doc_id` to the existing `ChunkMeta.doc_ids` list.

## Troubleshooting

- **Connection refused to the socket**: make sure `daser.server` is running
  before sending demo requests, and that vLLM uses the same `socket_path`.
- **`doc N has no cached tokens for prompt rebuild`**: a doc must be
  uploaded through `/documents` before `/infer` can use it; inferring
  against a doc that was evicted requires re-uploading.
- **vLLM rejects `max_tokens=0`**: the service uses `max_tokens=1` for
  prefill and discards the single decoded token.
