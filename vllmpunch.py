#!/usr/bin/env python3
"""
vllmpunch — configure named vLLM models and spawn them via podman (or docker).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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
        print(f"{name}\t{mid}")
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
    data["models"][name] = entry
    save_json(path, data)
    print(f"Added model '{name}' -> {args.model_id} in {path}")
    return 0


def merge_launch(launch: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    out = dict(launch)
    for k in (
        "host_port",
        "shm_size",
        "tensor_parallel_size",
        "container_name",
        "cache_dir",
        "api_host",
    ):
        if k in model and model[k] is not None:
            out[k] = model[k]
    return out


def build_vllm_argv(merged: dict[str, Any], model: dict[str, Any]) -> list[str]:
    """Arguments passed to the container image (vLLM entrypoint), after the image name."""
    model_id = model["model_id"]
    tp = int(merged.get("tensor_parallel_size", 1))
    argv = ["--model", model_id, "--tensor-parallel-size", str(tp)]
    extra_vllm = merged.get("extra_vllm_args") or []
    if isinstance(extra_vllm, list):
        argv.extend(str(x) for x in extra_vllm)
    return argv


def echo_run_commands(podman_argv: list[str], vllm_argv: list[str]) -> None:
    """Print the vLLM args and full podman invocation (shell-quoted) to stderr."""
    print(f"vllm: {shlex.join(vllm_argv)}", file=sys.stderr)
    print(f"podman: {shlex.join(podman_argv)}", file=sys.stderr)


def build_podman_argv(launch: dict[str, Any], model: dict[str, Any]) -> list[str]:
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

    argv: list[str] = [runtime, "run", "--rm", "-it", *[str(x) for x in extra_podman]]
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


def cmd_run(args: argparse.Namespace) -> int:
    models_path = resolve_models_config(args.models_config)
    launch_path = resolve_launch_config(args.launch_config)
    models_data = load_json(models_path)
    launch_data = load_json(launch_path)
    models = models_data.get("models") or {}
    name = args.model
    if name not in models:
        print(f"Unknown model '{name}'. Use 'list' or 'add'. Config: {models_path}", file=sys.stderr)
        return 1
    model_entry = models[name]
    if "model_id" not in model_entry:
        print(f"Model '{name}' has no model_id.", file=sys.stderr)
        return 1
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
    argv = build_podman_argv(launch_merged, model_entry)
    if args.echo_command:
        echo_run_commands(argv, vllm_argv)
    if args.dry_run:
        print(json.dumps(argv))
        return 0
    os.execvp(argv[0], argv)


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
    name = args.model
    if name not in models:
        print(f"Unknown model '{name}'. Use 'list' or 'add'. Config: {models_path}", file=sys.stderr)
        return 1
    model_entry = models[name]
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
        help="Models JSON (default: ./vllmpunch-models.json if present else ~/.config/vllmpunch/models.json)",
    )
    common.add_argument(
        "--launch-config",
        type=Path,
        metavar="PATH",
        help="Launch options JSON (default: ./vllmpunch-launch.json if present else ~/.config/vllmpunch/launch.json)",
    )

    p = argparse.ArgumentParser(
        prog="vllmpunch",
        description="List/add vLLM model aliases and run podman with merged launch settings.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "list",
        parents=[common],
        help="List configured model names and Hugging Face ids",
    )

    ap_add = sub.add_parser("add", parents=[common], help="Add a named model to the models config")
    ap_add.add_argument("name", help="Short name for this model")
    ap_add.add_argument("model_id", help="Hugging Face model id, e.g. org/model")
    ap_add.add_argument("--cache-dir", help="Host cache directory to mount (default: from launch or ./rhaiis-cache at run time)")
    ap_add.add_argument("--host-port", type=int, help="Host port mapped to vLLM in the container")
    ap_add.add_argument("--shm-size", help="e.g. 4g or 6g")
    ap_add.add_argument("--container-name", help="Optional podman --name")
    ap_add.add_argument("--tensor-parallel-size", type=int, help="Override tensor parallel size for this model")

    ap_run = sub.add_parser("run", parents=[common], help="Spawn vLLM for a named model")
    ap_run.add_argument("model", help="Name from the models config")
    ap_run.add_argument(
        "-e",
        "--echo-command",
        action="store_true",
        dest="echo_command",
        help="Echo the vLLM args and full podman line (shell-quoted) to stderr, then run",
    )
    ap_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print podman argv as JSON instead of executing",
    )

    ap_prompt = sub.add_parser(
        "prompt",
        parents=[common],
        help="Interactive chat against a running vLLM HTTP API (second terminal; uses model host_port)",
    )
    ap_prompt.add_argument("model", help="Name from the models config (must match the running server)")
    ap_prompt.add_argument(
        "--host",
        dest="api_host",
        metavar="ADDR",
        help="API host (default: api_host from config, or $VLLMPUNCH_API_HOST, or 127.0.0.1)",
    )
    ap_prompt.add_argument(
        "--base-url",
        metavar="URL",
        help="Override full HTTP base, e.g. http://127.0.0.1:8002 (skips host/port from config)",
    )
    ap_prompt.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for each completion (default: 120)",
    )

    args_ns = p.parse_args()

    if args_ns.command == "list":
        return cmd_list(args_ns)
    if args_ns.command == "add":
        return cmd_add(args_ns)
    if args_ns.command == "run":
        return cmd_run(args_ns)
    if args_ns.command == "prompt":
        return cmd_prompt(args_ns)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
