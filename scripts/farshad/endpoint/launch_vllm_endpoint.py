#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import re
from typing import Any, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

try:
    import yaml  # type: ignore
except Exception as e:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise

# Local utils
from vllm_endpoint_utils import (
    validate_config_schema,
    build_vllm_command,
    extract_host_port,
    validate_paths,
    is_port_available,
    compute_probe_hosts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch a vLLM endpoint from YAML config")
    p.add_argument("--config", "-c", required=True, help="Path to endpoint config YAML")
    p.add_argument(
        "--endpoint-name",
        default="vllm",
        help="Endpoint name to use from the YAML 'endpoint' list (default: vllm)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and environment without launching",
    )
    p.add_argument(
        "--print-env",
        action="store_true",
        help="Print environment variables that will be set",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Optional file to redirect server stdout/stderr",
    )
    p.add_argument(
        "--wait-ready",
        action="store_true",
        help="Wait for the endpoint /health to return 200 before exiting",
    )
    p.add_argument(
        "--wait-timeout",
        type=int,
        default=600,
        help="Max seconds to wait for health check (default: 600)",
    )
    p.add_argument(
        "--wait-interval",
        type=float,
        default=2.0,
        help="Seconds between health probes (default: 2.0)",
    )
    p.add_argument(
        "--health-endpoint",
        default="/health",
        help="Health endpoint path (default: /health)",
    )
    p.add_argument(
        "--skip-path-checks",
        action="store_true",
        help="Skip filesystem path validation for model/chat-template/etc.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load YAML
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate schema
    try:
        validate_config_schema(config)
    except Exception as e:
        print(f"Config validation error: {e}", file=sys.stderr)
        return 2

    # Select endpoint by name
    endpoints = config["endpoint"]
    selected: Optional[Dict[str, Any]] = None
    for ep in endpoints:
        if str(ep.get("name", "")) == args.endpoint_name:
            selected = ep
            break
    if selected is None:
        print(
            f"No endpoint with name '{args.endpoint_name}' found in config",
            file=sys.stderr,
        )
        return 3

    # Optional path validation
    base_dir = os.path.dirname(os.path.abspath(args.config))
    if not args.skip_path_checks:
        path_errors = validate_paths(selected, base_dir)
        if path_errors:
            for e in path_errors:
                print(f"Path validation error: {e}", file=sys.stderr)
            return 6

    # Build command and environment
    try:
        cmd, env_vars = build_vllm_command(selected, config.get("devices"))
    except Exception as e:
        print(f"Error building command: {e}", file=sys.stderr)
        return 4
    # Derive default log file from config name and platform if not provided
    def _slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-_")

    cfg_name = str(config.get("name", "endpoint"))
    cfg_platform = str(config.get("platform", "platform"))
    derived_log_file = f"vllm_serve_{_slug(cfg_name)}_{_slug(cfg_platform)}.log"
    resolved_log_file = args.log_file or derived_log_file
    logger.info(f"Devices specified: {config.get('devices', 'None')}")
    # Show result if dry-run (skip port/path network checks)
    if args.dry_run:
        print("Command:", " ".join(cmd))
        if args.print_env and env_vars:
            print("Environment:")
            for k, v in env_vars.items():
                print(f"  {k}={v}")
            # Also show effective ZE_AFFINITY_MASK if set via environment or top-level devices
            eff = os.environ.get("ZE_AFFINITY_MASK") or (str(config.get("devices", "")).strip() or None)
            if eff:
                print(f"  ZE_AFFINITY_MASK={eff}")
        
        return 0

    # Check port availability on Linux before launching
    host, port = extract_host_port(selected)
    if not is_port_available(host, port):
        print(
            f"Error: Port {port} on host {host} is already in use. Please choose a different port.",
            file=sys.stderr,
        )
        return 8

    # Prepare environment
    launch_env = os.environ.copy()
    launch_env.update({k: str(v) for k, v in env_vars.items()})

    # Endpoint-agnostic affinity: prefer existing env, otherwise top-level devices.
    top_devices = str(config.get("devices", "")).strip() or None
    # For xpu, set ZE_AFFINITY_MASK; for cpu, set CPU_VISIBLE_MEMORY_NODES (if not already set)
    if config.get("platform") == "xpu":
        if "ZE_AFFINITY_MASK" not in launch_env and top_devices:
            launch_env["ZE_AFFINITY_MASK"] = top_devices
            os.environ["ZE_AFFINITY_MASK"] = top_devices  # For subprocess visibility
            logger.info(f"Setting ZE_AFFINITY_MASK={top_devices}")
        elif "ZE_AFFINITY_MASK" in launch_env:
            logger.info(f"Using existing ZE_AFFINITY_MASK={launch_env['ZE_AFFINITY_MASK']}")
    else:
        if "CPU_VISIBLE_MEMORY_NODES" not in launch_env and top_devices:
            launch_env["CPU_VISIBLE_MEMORY_NODES"] = top_devices
            os.environ["CPU_VISIBLE_MEMORY_NODES"] = top_devices  # For subprocess visibility
            logger.info(f"Setting CPU_VISIBLE_MEMORY_NODES={top_devices}")
        elif "CPU_VISIBLE_MEMORY_NODES" in launch_env:
            logger.info(f"Using existing CPU_VISIBLE_MEMORY_NODES={launch_env['CPU_VISIBLE_MEMORY_NODES']}")

    # Launch
    stdout = None
    stderr = None
    if resolved_log_file:
        log_fh = open(resolved_log_file, "wb", buffering=0)
        stdout = log_fh
        stderr = subprocess.STDOUT
        logger.info(f"Redirecting output to {resolved_log_file}")

    logger.info("Launching: " + " ".join(cmd))
    try:
        proc = subprocess.Popen(cmd, env=launch_env, stdout=stdout, stderr=stderr)
    except FileNotFoundError:
        logger.error(
            "Error: 'vllm' command not found. Ensure vLLM is installed and in PATH."
        )
        return 5

    logger.info(f"vLLM server started with PID {proc.pid}")

    # Optional wait for readiness
    if args.wait_ready:
        bind_host, port = extract_host_port(selected)
        probe_hosts = compute_probe_hosts(bind_host)
        import time
        import urllib.request
        import urllib.error

        deadline = time.time() + float(args.wait_timeout)
        last_err = None
        logger.info(f"Waiting for readiness (bind={probe_hosts}, port={port}) ...")
        while time.time() < deadline:
            for ph in probe_hosts:
                url = f"http://{ph}:{port}{args.health_endpoint}"
                try:
                    with urllib.request.urlopen(url, timeout=5) as resp:
                        code = getattr(resp, 'status', None) or resp.getcode()
                        if code == 200:
                            logger.info(f"Endpoint is healthy at {url} (200).")
                            logger.info("Tip: use --dry-run to preview the command.")
                            return 0
                except Exception as ex:  # noqa: BLE001 (broad for readiness loop)
                    last_err = ex
            time.sleep(float(args.wait_interval))

        print("Health check timed out.", file=sys.stderr)
        if last_err:
            print(f"Last error: {last_err}", file=sys.stderr)
        return 7

    print("Tip: use --dry-run to preview the command.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
