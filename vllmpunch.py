#!/usr/bin/env python3
"""
vllmpunch — configure named vLLM models and spawn them via podman (or docker).

Full documentation: README.md next to this file. CLI reference: vllmpunch --help
and vllmpunch <command> --help.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Preserve description/epilog whitespace for multi-line help."""


_VLLMPUNCH_EPILOG = """\
config files:
  models   ./vllmpunch-models.json if present, else ~/.config/vllmpunch/models.json
  launch   ./vllmpunch-launch.json if present, else ~/.config/vllmpunch/launch.json
  override with --models-config and --launch-config

model entry fields (JSON):
  model_id, alias, aliases, host_port, container_port, cache_dir, shm_size,
  container_name, tensor_parallel_size, vllm_flags, extra_vllm_args

vllm_flags keys -> vLLM CLI:
  serve_port -> --port
  max_model_len -> --max-model-len
  gpu_memory_utilization -> --gpu-memory-utilization
  kv_cache_dtype -> --kv-cache-dtype
  quantization -> --quantization
  trust_remote_code (true) -> --trust-remote-code
  enforce_eager (true) -> --enforce-eager

environment:
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN  passed to container when set
  VLLMPUNCH_API_HOST                 default API host for prompt
  XDG_CONFIG_HOME                    base for ~/.config paths

See README.md in the vllmpunch directory for full documentation.
"""


def default_models_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "vllmpunch" / "models.json"


def default_launch_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "vllmpunch" / "launch.json"


def cwd_models_path() -> Path:
    return Path.cwd() / "vllmpunch-models.json"


def cwd_launch_path() -> Path:
    return Path.cwd() / "vllmpunch-launch.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def resolve_models_config(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if cwd_models_path().is_file():
        return cwd_models_path()
    return default_models_path()


def resolve_launch_config(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if cwd_launch_path().is_file():
        return cwd_launch_path()
    return default_launch_path()


def resolve_model_entry(models: dict[str, Any], name: str) -> tuple[str, dict[str, Any]] | None:
    """Resolve canonical model key and entry by top-level name, alias, or aliases."""
    if name in models:
        return name, models[name]
    for canonical in sorted(models.keys()):
        entry = models[canonical]
        if entry.get("alias") == name:
            return canonical, entry
        als = entry.get("aliases")
        if isinstance(als, list) and name in als:
            return canonical, entry
    return None


def format_aliases(entry: dict[str, Any]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    a = entry.get("alias")
    if isinstance(a, str) and a:
        parts.append(a)
        seen.add(a)
    als = entry.get("aliases")
    if isinstance(als, list):
        for x in als:
            s = str(x)
            if s and s not in seen:
                parts.append(s)
                seen.add(s)
    return ",".join(parts)


def cmd_list(args: argparse.Namespace) -> int:
    path = resolve_models_config(args.models_config)
    data = load_json(path)
    models = data.get("models") or {}
    if not models:
        print(f"No models in {path} (copy models.example.json or use 'add').", file=sys.stderr)
        return 1
    for name in sorted(models.keys()):
        entry = models[name]
        mid = entry.get("model_id", "?")
        aliases = format_aliases(entry)
        print(f"{name}\t{aliases}\t{mid}")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    path = resolve_models_config(args.models_config)
    data = load_json(path)
    if "models" not in data or not isinstance(data["models"], dict):
        data["models"] = {}
    name = args.name
    if name in data["models"]:
        print(f"Model '{name}' already exists. Remove it first or pick another name.", file=sys.stderr)
        return 1
    entry: dict[str, Any] = {"model_id": args.model_id}
    if args.cache_dir:
        entry["cache_dir"] = args.cache_dir
    if args.host_port is not None:
        entry["host_port"] = args.host_port
    if args.shm_size:
        entry["shm_size"] = args.shm_size
    if args.container_name:
        entry["container_name"] = args.container_name
    if args.tensor_parallel_size is not None:
        entry["tensor_parallel_size"] = args.tensor_parallel_size
    if getattr(args, "alias", None):
        ali = args.alias
        if len(ali) == 1:
            entry["alias"] = ali[0]
        else:
            entry["aliases"] = ali
    data["models"][name] = entry
    save_json(path, data)
    print(f"Added model '{name}' -> {args.model_id} in {path}")
    return 0


def merge_launch(launch: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    out = dict(launch)
    for k in (
        "host_port",
        "container_port",
        "shm_size",
        "tensor_parallel_size",
        "container_name",
        "cache_dir",
        "api_host",
    ):
        if k in model and model[k] is not None:
            out[k] = model[k]
    launch_extra = launch.get("extra_vllm_args") or []
    model_extra = model.get("extra_vllm_args") or []
    if isinstance(launch_extra, list) and isinstance(model_extra, list):
        out["extra_vllm_args"] = list(launch_extra) + list(model_extra)
    elif isinstance(model_extra, list) and model_extra:
        out["extra_vllm_args"] = list(model_extra)
    elif isinstance(launch_extra, list):
        out["extra_vllm_args"] = list(launch_extra)
    return out


def expand_vllm_flags(flags: Any) -> list[str]:
    """Convert per-model vllm_flags dict to vLLM CLI argv fragments."""
    if not isinstance(flags, dict):
        return []
    argv: list[str] = []
    if flags.get("serve_port") is not None:
        argv.extend(["--port", str(flags["serve_port"])])
    if flags.get("max_model_len") is not None:
        argv.extend(["--max-model-len", str(flags["max_model_len"])])
    if flags.get("gpu_memory_utilization") is not None:
        argv.extend(["--gpu-memory-utilization", str(flags["gpu_memory_utilization"])])
    if flags.get("kv_cache_dtype") is not None:
        argv.extend(["--kv-cache-dtype", str(flags["kv_cache_dtype"])])
    if flags.get("quantization") is not None:
        argv.extend(["--quantization", str(flags["quantization"])])
    if flags.get("trust_remote_code") is True:
        argv.append("--trust-remote-code")
    if flags.get("enforce_eager") is True:
        argv.append("--enforce-eager")
    return argv


def build_vllm_argv(merged: dict[str, Any], model: dict[str, Any]) -> list[str]:
    """Arguments passed to the container image (vLLM entrypoint), after the image name."""
    model_id = model["model_id"]
    tp = int(merged.get("tensor_parallel_size", 1))
    argv = ["--model", model_id, "--tensor-parallel-size", str(tp)]
    argv.extend(expand_vllm_flags(model.get("vllm_flags")))
    extra_vllm = merged.get("extra_vllm_args") or []
    if isinstance(extra_vllm, list):
        argv.extend(str(x) for x in extra_vllm)
    return argv


def echo_run_commands(podman_argv: list[str], vllm_argv: list[str]) -> None:
    """Print the vLLM args and full podman invocation (shell-quoted) to stderr."""
    print(f"vllm: {shlex.join(vllm_argv)}", file=sys.stderr)
    print(f"podman: {shlex.join(podman_argv)}", file=sys.stderr)


def build_podman_argv(
    launch: dict[str, Any],
    model: dict[str, Any],
    *,
    detach: bool = False,
) -> list[str]:
    merged = merge_launch(launch, model)
    runtime = merged.get("runtime") or "podman"
    image = merged["vllm_image"]
    host_port = int(merged.get("host_port", 8001))
    cport = int(merged.get("container_port", 8000))
    shm = merged.get("shm_size", "4g")
    device = merged.get("device", "nvidia.com/gpu=all")
    sec = merged.get("security_opt", "label=disable")
    userns = merged.get("userns", "keep-id:uid=1001")
    hf_off = str(merged.get("hf_hub_offline", "0"))
    cache_dir = merged.get("cache_dir") or "./rhaiis-cache"
    mount_target = merged.get("cache_mount_target", "/opt/app-root/src/.cache:Z")
    cache_path = Path(cache_dir).expanduser().resolve()

    extra_podman = merged.get("extra_podman_args") or []
    if not isinstance(extra_podman, list):
        extra_podman = []

    run_flags: list[str] = ["--rm", "-d" if detach else "-it"]
    argv: list[str] = [runtime, "run", *run_flags, *[str(x) for x in extra_podman]]
    cname = merged.get("container_name")
    if cname:
        argv.extend(["--name", str(cname)])
    argv.extend(
        [
            "--device",
            device,
            f"--security-opt={sec}",
            f"--shm-size={shm}",
            "-p",
            f"{host_port}:{cport}",
            f"--userns={userns}",
        ]
    )
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token is not None:
        argv.extend(["--env", f"HUGGING_FACE_HUB_TOKEN={hf_token}"])
    argv.extend(["--env", f"HF_HUB_OFFLINE={hf_off}"])
    argv.extend(["-v", f"{cache_path}:{mount_target}", image])
    argv.extend(build_vllm_argv(merged, model))
    return argv


def build_run_argv(
    models_path: Path,
    launch_path: Path,
    model_name: str,
    *,
    detach: bool,
) -> tuple[list[str], list[str], str] | tuple[None, None, None]:
    """Returns (podman_argv, vllm_argv, canonical_name) or (None, None, None) on error."""
    models_data = load_json(models_path)
    launch_data = load_json(launch_path)
    models = models_data.get("models") or {}
    resolved = resolve_model_entry(models, model_name)
    if resolved is None:
        print(f"Unknown model '{model_name}'. Use 'list' or 'add'. Config: {models_path}", file=sys.stderr)
        return None, None, None
    name, model_entry = resolved
    if "model_id" not in model_entry:
        print(f"Model '{name}' has no model_id.", file=sys.stderr)
        return None, None, None
    defaults: dict[str, Any] = {
        "runtime": "podman",
        "vllm_image": "registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.5",
        "device": "nvidia.com/gpu=all",
        "security_opt": "label=disable",
        "container_port": 8000,
        "tensor_parallel_size": 1,
        "userns": "keep-id:uid=1001",
        "hf_hub_offline": "0",
        "cache_mount_target": "/opt/app-root/src/.cache:Z",
    }
    launch_merged = {**defaults, **launch_data}
    merged = merge_launch(launch_merged, model_entry)
    vllm_argv = build_vllm_argv(merged, model_entry)
    argv = build_podman_argv(launch_merged, model_entry, detach=detach)
    return argv, vllm_argv, name


def cmd_run(args: argparse.Namespace) -> int:
    models_path = resolve_models_config(args.models_config)
    launch_path = resolve_launch_config(args.launch_config)
    argv, vllm_argv, _name = build_run_argv(
        models_path,
        launch_path,
        args.model,
        detach=args.detach,
    )
    if argv is None:
        return 1
    if args.echo_command:
        echo_run_commands(argv, vllm_argv)
    if args.dry_run:
        print(json.dumps(argv))
        return 0
    if args.detach:
        proc = subprocess.run(argv, check=False, capture_output=True, text=True)
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        return proc.returncode
    os.execvp(argv[0], argv)


def cmd_run_parallel(args: argparse.Namespace) -> int:
    """Start multiple models as detached containers (for concurrent agents on distinct ports)."""
    models_path = resolve_models_config(args.models_config)
    launch_path = resolve_launch_config(args.launch_config)
    rc = 0
    for model_name in args.models:
        argv, vllm_argv, name = build_run_argv(
            models_path,
            launch_path,
            model_name,
            detach=True,
        )
        if argv is None:
            rc = 1
            continue
        if args.echo_command:
            print(f"# {name}", file=sys.stderr)
            echo_run_commands(argv, vllm_argv)
        if args.dry_run:
            print(json.dumps({"model": name, "argv": argv}))
            continue
        proc = subprocess.run(argv, check=False, capture_output=True, text=True)
        out = proc.stdout.strip()
        cid = out.split("\n")[-1].strip() if out else ""
        if proc.returncode != 0:
            rc = 1
            err = proc.stderr.strip() or "(no stderr)"
            print(f"Failed to start {name}: {err}", file=sys.stderr)
            continue
        print(f"{name}\t{cid}")
    return rc


def resolve_api_base(merged: dict[str, Any], args: argparse.Namespace) -> str:
    """HTTP base for the OpenAI-compatible API (vLLM default: /v1/...)."""
    if getattr(args, "base_url", None):
        return str(args.base_url).rstrip("/")
    host = (
        getattr(args, "api_host", None)
        or merged.get("api_host")
        or os.environ.get("VLLMPUNCH_API_HOST")
        or "127.0.0.1"
    )
    port = int(merged.get("host_port", 8001))
    return f"http://{host}:{port}"


def chat_completion(
    api_base: str,
    model_id: str,
    messages: list[dict[str, str]],
    timeout: float,
) -> str:
    url = f"{api_base}/v1/chat/completions"
    body = json.dumps(
        {"model": model_id, "messages": messages, "stream": False, "temperature": 0.7}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"unexpected response: {data!r}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise ValueError(f"missing content in response: {data!r}")
    return str(content)


def cmd_prompt(args: argparse.Namespace) -> int:
    """Interactive REPL against a running vLLM server's OpenAI-compatible HTTP API."""
    models_path = resolve_models_config(args.models_config)
    launch_path = resolve_launch_config(args.launch_config)
    models_data = load_json(models_path)
    launch_data = load_json(launch_path)
    models = models_data.get("models") or {}
    resolved = resolve_model_entry(models, args.model)
    if resolved is None:
        print(f"Unknown model '{args.model}'. Use 'list' or 'add'. Config: {models_path}", file=sys.stderr)
        return 1
    name, model_entry = resolved
    if "model_id" not in model_entry:
        print(f"Model '{name}' has no model_id.", file=sys.stderr)
        return 1
    defaults: dict[str, Any] = {
        "host_port": 8001,
        "api_host": "127.0.0.1",
    }
    launch_merged = {**defaults, **launch_data}
    merged = merge_launch(launch_merged, model_entry)
    api_base = resolve_api_base(merged, args)
    model_id = model_entry["model_id"]
    timeout = float(getattr(args, "timeout", 120.0) or 120.0)

    print(
        f"Connected to {api_base}/v1 (model={model_id}). "
        "Type a message, empty line to quit, Ctrl-D EOF.",
        file=sys.stderr,
    )

    history: list[dict[str, str]] = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            print(file=sys.stderr)
            break
        if not line.strip():
            break
        history.append({"role": "user", "content": line})
        try:
            reply = chat_completion(api_base, model_id, history, timeout=timeout)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"HTTP {e.code}: {err_body}", file=sys.stderr)
            history.pop()
            continue
        except urllib.error.URLError as e:
            print(f"Connection failed ({api_base}): {e.reason}", file=sys.stderr)
            history.pop()
            continue
        except (TimeoutError, ValueError) as e:
            print(str(e), file=sys.stderr)
            history.pop()
            continue
        print(reply)
        history.append({"role": "assistant", "content": reply})

    return 0


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--models-config",
        type=Path,
        metavar="PATH",
        help=(
            "Path to models JSON (default: ./vllmpunch-models.json if present in cwd, "
            "else ~/.config/vllmpunch/models.json)"
        ),
    )
    common.add_argument(
        "--launch-config",
        type=Path,
        metavar="PATH",
        help="Path to launch/runtime JSON merged with each model (default: ./vllmpunch-launch.json "
        "if present, else ~/.config/vllmpunch/launch.json)",
    )

    p = argparse.ArgumentParser(
        prog="vllmpunch",
        formatter_class=_HelpFormatter,
        description=(
            "Configure named Hugging Face models and vLLM flags in JSON, then spawn "
            "podman/docker with merged launch settings. Merges launch.json with each model "
            "(host_port, container_port, shm_size, cache, extra_vllm_args, etc.)."
        ),
        epilog=_VLLMPUNCH_EPILOG,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    sub.add_parser(
        "list",
        parents=[common],
        formatter_class=_HelpFormatter,
        description=(
            "Print all configured models as tab-separated columns: "
            "canonical_name, aliases (comma-separated), model_id."
        ),
        help="List canonical names, aliases, and Hugging Face model ids",
    )

    ap_add = sub.add_parser(
        "add",
        parents=[common],
        formatter_class=_HelpFormatter,
        description=(
            "Append a new entry to the models JSON. Only basic fields can be set from the CLI; "
            "edit the file to add vllm_flags or extra_vllm_args."
        ),
        help="Add a named model (model_id and optional podman fields)",
    )
    ap_add.add_argument(
        "name",
        help="Canonical key in the models map (use quotes if the name contains special characters)",
    )
    ap_add.add_argument("model_id", help="Hugging Face repository id, e.g. org/model")
    ap_add.add_argument(
        "--cache-dir",
        metavar="DIR",
        help="Host directory mounted as the HF cache inside the container",
    )
    ap_add.add_argument(
        "--host-port",
        type=int,
        metavar="PORT",
        help="Published host TCP port for the OpenAI-compatible HTTP API",
    )
    ap_add.add_argument(
        "--shm-size",
        metavar="SIZE",
        help="Podman --shm-size (e.g. 4g, 6g)",
    )
    ap_add.add_argument(
        "--container-name",
        metavar="NAME",
        help="Podman --name for this server container",
    )
    ap_add.add_argument(
        "--tensor-parallel-size",
        type=int,
        metavar="N",
        help="vLLM --tensor-parallel-size for this model",
    )
    ap_add.add_argument(
        "--alias",
        action="append",
        metavar="NAME",
        dest="alias",
        help=(
            "Shortcut for run/prompt/list: use once to set 'alias'; "
            "repeat to store multiple values as 'aliases'"
        ),
    )

    ap_run = sub.add_parser(
        "run",
        parents=[common],
        formatter_class=_HelpFormatter,
        description=(
            "Start one vLLM container for MODEL (canonical name or alias). "
            "Without --detach, replaces this process with podman (foreground, -it). "
            "With --detach, runs podman in the background and prints the container id."
        ),
        help="Run a single model (foreground unless --detach)",
    )
    ap_run.add_argument(
        "model",
        metavar="MODEL",
        help="Model entry key or alias from the models config",
    )
    ap_run.add_argument(
        "-e",
        "--echo-command",
        action="store_true",
        dest="echo_command",
        help=(
            "Print 'vllm: …' and 'podman: …' (shell-quoted argv) to stderr, "
            "then execute or dry-run"
        ),
    )
    ap_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the podman argv as a JSON array and exit without starting a container",
    )
    ap_run.add_argument(
        "--detach",
        "-d",
        action="store_true",
        help="podman run -d (detached) instead of -it; stdout gets the container id",
    )

    ap_rp = sub.add_parser(
        "run-parallel",
        parents=[common],
        formatter_class=_HelpFormatter,
        description=(
            "Start several models in sequence, each as a detached container (same as 'run -d' per model). "
            "Requires distinct host_port (and matching container_port/serve_port when using --port). "
            "Avoid starting many large models at once on one GPU (OOM risk)."
        ),
        help="Start multiple models as detached containers (sequential)",
    )
    ap_rp.add_argument(
        "models",
        nargs="+",
        metavar="MODEL",
        help="One or more canonical names or aliases, e.g. llama qwen-coder nemotron-nano",
    )
    ap_rp.add_argument(
        "-e",
        "--echo-command",
        action="store_true",
        dest="echo_command",
        help="Before each start, print vllm and podman lines for that model to stderr",
    )
    ap_rp.add_argument(
        "--dry-run",
        action="store_true",
        help="For each model, print one JSON object with keys 'model' and 'argv'; no execution",
    )

    ap_prompt = sub.add_parser(
        "prompt",
        parents=[common],
        formatter_class=_HelpFormatter,
        description=(
            "Interactive stdin/stdout chat against a running server's OpenAI-compatible API "
            "(/v1/chat/completions). The server must already be running; connection URL is built "
            "from merged api_host and host_port unless --base-url is set."
        ),
        help="REPL chat against a running vLLM HTTP API",
    )
    ap_prompt.add_argument(
        "model",
        metavar="MODEL",
        help="Same model entry as the running container (canonical name or alias)",
    )
    ap_prompt.add_argument(
        "--host",
        dest="api_host",
        metavar="ADDR",
        help="Hostname for http://HOST:PORT (overridden by --base-url)",
    )
    ap_prompt.add_argument(
        "--base-url",
        metavar="URL",
        help="Full API root (e.g. http://127.0.0.1:8002); overrides host and port from config",
    )
    ap_prompt.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        metavar="SEC",
        help="HTTP timeout for each completion request (default: 120)",
    )

    args_ns = p.parse_args()

    if args_ns.command == "list":
        return cmd_list(args_ns)
    if args_ns.command == "add":
        return cmd_add(args_ns)
    if args_ns.command == "run":
        return cmd_run(args_ns)
    if args_ns.command == "run-parallel":
        return cmd_run_parallel(args_ns)
    if args_ns.command == "prompt":
        return cmd_prompt(args_ns)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
