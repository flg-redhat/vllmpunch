# vllmpunch

**vllmpunch** is a small CLI that keeps named Hugging Face models, vLLM flags, and Podman/Docker launch settings in JSON. It builds the `podman run …` line (or `docker`) and the vLLM arguments passed into the container image.

## Quick start

From this directory:

```bash
./vllmpunch list
./vllmpunch run -e llama                    # foreground (interactive TTY)
./vllmpunch run -d qwen-coder               # background detached container
./vllmpunch run-parallel orchestrator qwen-coder nemotron-nano
```

Use `-e` / `--echo-command` to print the exact **vllm** and **podman** argument lists before execution.

## Configuration paths

| File | Resolution |
|------|------------|
| Models | `./vllmpunch-models.json` if it exists in the current working directory; otherwise `$XDG_CONFIG_HOME/vllmpunch/models.json` (typically `~/.config/vllmpunch/models.json`). |
| Launch | `./vllmpunch-launch.json` if present; else `~/.config/vllmpunch/launch.json`. |

Override with global CLI options:

- `--models-config PATH`
- `--launch-config PATH`

Examples ship as [`models.example.json`](models.example.json) and [`launch.example.json`](launch.example.json).

## Models JSON (`models`)

Top-level object with a `models` map. Each **key** is the canonical model name; the value is an object that may include:

| Field | Purpose |
|-------|---------|
| `model_id` | Hugging Face model id (required for `run` / `prompt`). |
| `alias` | Single short name for `run`, `prompt`, `list`. |
| `aliases` | List of additional short names. |
| `host_port` | Host TCP port published to the container (OpenAI-compatible HTTP API on the host). |
| `container_port` | Port vLLM listens on **inside** the container. Defaults from launch config (often `8000`). Set this when you use `vllm_flags.serve_port` so `-p host:container` matches `--port` inside the container. |
| `cache_dir` | Host directory mounted as the Hugging Face cache inside the container. |
| `shm_size` | Podman `--shm-size` (e.g. `4g`). |
| `container_name` | Podman `--name`. |
| `tensor_parallel_size` | Passed to vLLM as `--tensor-parallel-size`. |
| `vllm_flags` | Structured vLLM options (see below). |
| `extra_vllm_args` | Extra strings appended to the vLLM argv after `vllm_flags` (merged with launch `extra_vllm_args`). |

### `vllm_flags` → CLI mapping

These keys under `vllm_flags` become dashed vLLM flags:

| JSON key | vLLM flag |
|------------|-----------|
| `serve_port` | `--port` |
| `max_model_len` | `--max-model-len` |
| `gpu_memory_utilization` | `--gpu-memory-utilization` |
| `kv_cache_dtype` | `--kv-cache-dtype` |
| `quantization` | `--quantization` |
| `trust_remote_code` | `--trust-remote-code` (only if `true`) |
| `enforce_eager` | `--enforce-eager` (only if `true`) |

Order after `--model` and `--tensor-parallel-size`: expanded `vllm_flags`, then merged `extra_vllm_args`.

### Merging launch and model

For each run, vllmpunch merges **launch.json** defaults with the chosen **model** entry:

- Scalar overrides copied from the model: `host_port`, `container_port`, `shm_size`, `tensor_parallel_size`, `container_name`, `cache_dir`, `api_host`.
- `extra_vllm_args`: **launch list + model list** concatenated.

## Launch JSON (`launch`)

Shared Podman/Docker and runtime defaults. Common keys (see [`launch.example.json`](launch.example.json)):

| Field | Purpose |
|-------|---------|
| `runtime` | `podman` or `docker`. |
| `vllm_image` | Container image for vLLM. |
| `device` | GPU device spec (e.g. `nvidia.com/gpu=all`). |
| `security_opt`, `userns` | Passed through to the runtime. |
| `container_port` | Default inner port when the model does not set one. |
| `tensor_parallel_size` | Default tensor parallelism. |
| `hf_hub_offline` | Sets `HF_HUB_OFFLINE` in the container. |
| `cache_mount_target` | Volume mount target for the cache directory. |
| `extra_podman_args` | Extra strings inserted after `podman run --rm …`. |
| `extra_vllm_args` | Extra vLLM argv fragments merged with any model `extra_vllm_args`. |
| `api_host` | Default HTTP host for `prompt` / URL resolution. |

## Environment variables

| Variable | Effect |
|----------|--------|
| `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` | Passed into the container when set (model downloads). |
| `VLLMPUNCH_API_HOST` | Default API host for `prompt` if not set in config. |
| `XDG_CONFIG_HOME` | Base directory for default config paths. |

## Commands

### `list`

Prints tab-separated rows: **canonical name**, **aliases** (comma-separated), **model_id**.

### `add`

Adds a model entry. Positional arguments: `name`, `model_id`. Optional: `--cache-dir`, `--host-port`, `--shm-size`, `--container-name`, `--tensor-parallel-size`, `--alias` (repeatable; one value stores `alias`, multiple values store `aliases`).

### `run`

Starts one model. Positional: `model` (canonical name or alias).

| Option | Meaning |
|--------|---------|
| `-e`, `--echo-command` | Print **vllm:** and **podman:** lines (shell-quoted) to stderr, then proceed. |
| `--dry-run` | Print the podman argv as JSON and exit (no execution). |
| `-d`, `--detach` | Use `podman run -d` instead of foreground `-it`; prints container ID. |

Without `--detach`, vllmpunch **replaces itself** with `podman` (`execvp`), so logs stay attached to your terminal.

### `run-parallel`

Starts **multiple** models **in order**, each as a **detached** container (same as `run -d` per model). Use distinct `host_port` values per model so listeners do not collide.

Positional: one or more `MODEL` names or aliases.

| Option | Meaning |
|--------|---------|
| `-e`, `--echo-command` | For each model, print **vllm:** and **podman:** to stderr before starting. |
| `--dry-run` | Print one JSON object per line (`model`, `argv`) and exit. |

On success, prints `canonical_name<TAB>container_id` per line.

**VRAM:** Starting many heavy models at once can OOM the GPU. Prefer a strict startup sequence, clear stray GPU processes first (e.g. check `fuser -v /dev/nvidia0`), and tune `--gpu-memory-utilization` per model so the sum fits your card.

### `prompt`

Interactive REPL calling the OpenAI-compatible HTTP API (`/v1/chat/completions`) for the given model. Uses `host_port` and `api_host` from merged config unless overridden.

| Option | Meaning |
|--------|---------|
| `--host ADDR` | API hostname (default: config / `VLLMPUNCH_API_HOST` / `127.0.0.1`). |
| `--base-url URL` | Full base URL; overrides host and port from config. |
| `--timeout SEC` | Per-request timeout (default: 120). |

## Multi-agent example

A typical pattern is an orchestrator on port **8000**, a coder on **8001**, and a second specialist on **8002**, each with its own `host_port`, optional `container_port` / `serve_port`, and conservative `gpu_memory_utilization` values. Use `run-parallel orchestrator qwen-coder nemotron-nano` only after ports and GPU budgets are set in [`vllmpunch-models.json`](vllmpunch-models.json).

## License

See [LICENSE](LICENSE).
