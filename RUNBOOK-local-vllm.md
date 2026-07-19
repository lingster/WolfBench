# Local vLLM Smoke-Test Runbook

## Preflight Expectation

The smoke-test script probes the LLM server at `127.0.0.1:<port>/v1/models` locally before running benchmarks. Ensure the vLLM or Ollama server is bound to `0.0.0.0` and responding to OpenAI-compatible API calls on the specified port.

## Container→Host Networking

Docker containers (via harbor's docker sandbox) use **bridge networking** by default, not host networking. Containers reach the host via the special DNS name `host.docker.internal`, which resolves to the Docker daemon's host IP. This is standard Docker Desktop behavior and is supported on Linux with modern Docker versions (17.06+).

A process inside the task container should address a server bound on the host at port 8100 as:
```
http://10.13.1.10:8100/v1/models
```

This is handled automatically by `wolfbench-ollama.py` via the `_rewrite_api_base_for_docker()` function (lines 149–163 in wolfbench-ollama.py), which rewrites `localhost` and `127.0.0.1` URLs to `host.docker.internal` when using the docker sandbox.

## Smoke-Test Commands

### Model 1: gemma4-26b-int4-xr64

```bash
uv run wolfbench-ollama.py gemma4-26b-int4-xr64 \
  --api-base http://10.13.1.10:8100/v1 \
  --smoke \
  --skip-upload \
  --skip-chart \
  --sandbox docker
```

### Model 2: qwen3-30b-w4a16-xr64

```bash
uv run wolfbench-ollama.py qwen3-30b-w4a16-xr64 \
  --api-base http://10.13.1.10:8100/v1 \
  --smoke \
  --skip-upload \
  --skip-chart \
  --sandbox docker
```

## Notes

- Models are run one at a time; do not run both concurrently
- `--api-base http://10.13.1.10:8100/v1` points harbor's containers to the host vLLM server on port 8100
- `--smoke` runs only 1 task with a 600-second timeout (vs. full 89 tasks with 3600-second timeout)
- `--skip-upload` skips W&B upload (we already set dummy OPENAI_API_KEY in .env)
- `--skip-chart` skips HTML chart generation
- `--sandbox docker` uses local Docker containers (not cloud Daytona)
- Smoke runs are excluded from full leaderboard results but useful for validation

> **Linux note:** `host.docker.internal` does NOT resolve in containers on a
> plain Linux Docker engine (it is Docker Desktop behavior) and harbor's
> compose files do not add `host-gateway`. Use the host LAN IP (10.13.1.10)
> in `--api-base`; the script's localhost→host.docker.internal rewrite is
> bypassed when the host is already non-local. vLLM must bind 0.0.0.0.
