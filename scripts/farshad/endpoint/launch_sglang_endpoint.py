#!/usr/bin/env python3

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sglang_endpoint_utils import load_sglang_endpoint_config


def _env_int_from(env: Dict[str, str], name: str, default: Optional[int] = None) -> int:
    raw = env.get(name)
    if raw is None:
        if default is None:
            raise ValueError(f"Missing required environment variable: {name}")
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an SGLang endpoint (env-based or YAML config)"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to endpoint config YAML (sglang endpoint)",
    )
    parser.add_argument(
        "--endpoint-name",
        default="",
        help="Endpoint name to use from the YAML 'endpoint' list (default: first entry)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and environment without launching",
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print environment variables that will be set",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional file to redirect server stdout/stderr",
    )
    parser.add_argument(
        "--detach-router",
        action="store_true",
        help="Launch the router in the background and return immediately",
    )
    return parser.parse_args()


def _http_code(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return str(response.getcode())
    except urllib.error.HTTPError as exc:
        return str(exc.code)
    except Exception:
        return "000"


def _parse_numa_physical_core_range(node: int) -> Tuple[int, int]:
    result = subprocess.run(["lscpu"], capture_output=True, text=True, check=True)
    pattern = re.compile(rf"^NUMA node{node} CPU\(s\):\s*(.+)$", re.MULTILINE)
    match = pattern.search(result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse NUMA node{node} CPU(s) from lscpu output")

    cpu_spec = match.group(1).strip()
    first_segment = cpu_spec.split(",")[0].strip()
    if "-" in first_segment:
        start_str, end_str = first_segment.split("-", 1)
        return int(start_str), int(end_str)

    core = int(first_segment)
    return core, core


def _parse_lscpu_value(pattern: str, output: str, label: str) -> int:
    match = re.search(pattern, output, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse {label} from lscpu output")
    return int(match.group(1).strip())


def _get_lscpu_output() -> str:
    result = subprocess.run(["lscpu"], capture_output=True, text=True, check=True)
    return result.stdout


def _parse_numa_node_count(output: str) -> int:
    return _parse_lscpu_value(r"^NUMA node\(s\):\s*(\d+)\s*$", output, "NUMA node count")


def _parse_total_cpus(output: str) -> int:
    return _parse_lscpu_value(r"^CPU\(s\):\s*(\d+)\s*$", output, "CPU count")


def _extract_flag_value(args: List[str], *flag_names: str) -> Optional[str]:
    for i, arg in enumerate(args):
        if arg in flag_names:
            if i + 1 < len(args):
                return args[i + 1]
            return None
    return None


def _launch_server(
    port: int,
    cpu_affinity: str,
    numa_node: int,
    server_args: List[str],
    log_path: Path,
    host: str,
    env_overrides: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    print(
        f"Launching server on port {port} with CPU affinity {cpu_affinity} "
        f"(NUMA node {numa_node})"
    )

    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    if "SGLANG_CPU_OMP_THREADS_BIND" not in env:
        env["SGLANG_CPU_OMP_THREADS_BIND"] = cpu_affinity
    for key in ("NO_PROXY", "no_proxy"):
        current = env.get(key, "")
        if "0.0.0.0" not in current:
            env[key] = ",".join(filter(None, [current, "0.0.0.0"]))

    cmd = [
        "numactl",
        "-m",
        str(numa_node),
        "-C",
        cpu_affinity,
        "python3",
        "-m",
        "sglang.launch_server",
    ]
    cmd.extend(server_args)
    cmd.extend(["--numa-node", str(numa_node)])
    cmd.extend(["--host", host, "--port", str(port)])

    log_fh = open(log_path, "wb", buffering=0)
    return subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)


def _health_check_servers(url_list: List[str]) -> List[str]:
    print("Waiting for servers to start and checking health...")
    healthy: List[str] = []

    for url in url_list:
        print(f"Checking server at: {url}")
        max_attempts = 12
        attempt = 0
        server_ready = False

        while attempt < max_attempts:
            code = _http_code(f"{url}/health_generate", timeout=10)
            if code == "200":
                server_ready = True
                break

            if code == "000":
                root_code = _http_code(url, timeout=10)
                if root_code in {"200", "404", "405"}:
                    print("  ! /health_generate not ready, but server is reachable.")
                code = root_code

            print(
                f"  Attempt {attempt + 1}/{max_attempts}: Server not ready (HTTP {code}), waiting 5 seconds..."
            )
            time.sleep(5)
            attempt += 1

        if server_ready:
            print(f"  ✓ Server at {url} is reachable and healthy.")
            healthy.append(url)
        else:
            print(f"  ✗ Server at {url} is not reachable after {max_attempts} attempts.")

    print(f"Health check completed. {len(healthy)} out of {len(url_list)} servers are healthy.")
    return healthy


def _launch_servers(
    start_node: int,
    end_node_exclusive: int,
    total_instances: int,
    cores_per_inst: int,
    start_port: int,
    output_dir: Path,
    server_args: List[str],
    server_host: str,
    server_env: Dict[str, str],
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    url_list: List[str] = []
    num_launched_servers = 0

    for node in range(start_node, end_node_exclusive):
        start_core, end_core = _parse_numa_physical_core_range(node)

        while start_core <= end_core and num_launched_servers < total_instances:
            cpu_affinity = f"{start_core}-{start_core + cores_per_inst - 1}"
            print(f"Launching server on NUMA node {node} with CPU affinity {cpu_affinity}")

            port = start_port + num_launched_servers
            log_path = output_dir / f"server_{node}_{start_core}.log"
            _launch_server(
                port=port,
                cpu_affinity=cpu_affinity,
                numa_node=node,
                server_args=server_args,
                log_path=log_path,
                host=server_host,
                env_overrides=server_env,
            )

            probe_host = server_host
            if probe_host in {"0.0.0.0", "::"}:
                probe_host = "127.0.0.1"
            url_list.append(f"http://{probe_host}:{port}")
            time.sleep(5)

            start_core += cores_per_inst
            next_affinity_end = start_core + cores_per_inst - 1

            num_launched_servers += 1
            print(f"Number of launched servers: {num_launched_servers}")

            if next_affinity_end > end_core:
                break

        print(f"Finished launching servers on NUMA node {node}\n")

    return url_list


def main() -> int:
    args = parse_args()
    tic = int(time.time())

    config_env_vars: Dict[str, str] = {}
    config_server_cmd_args: List[str] = []
    config_host: Optional[str] = None
    config_seen_names: List[str] = []
    config_router_cmd_args: List[str] = []
    config_router_seen_names: List[str] = []

    try:
        (
            config_env_vars,
            config_server_cmd_args,
            config_host,
            config_seen_names,
            config_router_cmd_args,
            config_router_seen_names,
        ) = load_sglang_endpoint_config(args.config, args.endpoint_name)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Config validation error: {exc}", file=sys.stderr)
        return 2

    if "NUM_NUMA_NODES" in config_env_vars:
        print(
            "Unsupported env var NUM_NUMA_NODES. "
            "Use END_NUMA_NODE_INCLUSIVE instead.",
            file=sys.stderr,
        )
        return 2

    if "model-path" not in config_seen_names:
        print("Missing required cmd_arg: model-path", file=sys.stderr)
        return 2

    try:
        cores_per_inst = _env_int_from(config_env_vars, "CORES_PER_INST")
        start_port = _env_int_from(config_env_vars, "START_PORT")
        output_dir = Path(config_env_vars["OUTPUT_DIR"])
    except KeyError as exc:
        print(f"Missing required environment variable: {exc.args[0]}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    start_node = config_env_vars.get("START_NODE")
    end_numa_node_inclusive = config_env_vars.get("END_NUMA_NODE_INCLUSIVE")
    total_instances = config_env_vars.get("TOTAL_INSTANCES")

    lscpu_output: Optional[str] = None
    if start_node is None:
        start_node = 0
    else:
        try:
            start_node = int(start_node)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if end_numa_node_inclusive is None or total_instances is None:
        lscpu_output = _get_lscpu_output()

    if end_numa_node_inclusive is None:
        try:
            numa_node_count = _parse_numa_node_count(lscpu_output or "")
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
        end_numa_node_inclusive = start_node + numa_node_count - 1
    else:
        try:
            end_numa_node_inclusive = int(end_numa_node_inclusive)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if end_numa_node_inclusive < start_node:
        print("END_NUMA_NODE_INCLUSIVE must be >= START_NODE", file=sys.stderr)
        return 2

    end_node_exclusive = end_numa_node_inclusive + 1

    try:
        per_node_max_instances = {}
        total_capacity_instances = 0
        for node in range(start_node, end_node_exclusive):
            node_start, node_end = _parse_numa_physical_core_range(node)
            node_cores = node_end - node_start + 1
            max_instances = node_cores // cores_per_inst
            per_node_max_instances[node] = max_instances
            total_capacity_instances += max_instances
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if total_instances is None:
        total_instances = max(1, total_capacity_instances)
    else:
        try:
            total_instances = int(total_instances)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        if total_instances > total_capacity_instances:
            print(
                "Requested TOTAL_INSTANCES exceeds available CPU capacity; "
                f"trimming to {total_capacity_instances}.",
                file=sys.stderr,
            )
            total_instances = total_capacity_instances

    print(
        "Launch defaults: "
        + f"START_NODE={start_node}, "
        + f"END_NUMA_NODE_INCLUSIVE={end_numa_node_inclusive}, "
        + f"END_NODE_EXCLUSIVE={end_node_exclusive}, "
        + f"TOTAL_INSTANCES={total_instances}, "
        + f"CORES_PER_INST={cores_per_inst}, "
        + f"START_PORT={start_port}"
    )
    try:
        for node in range(start_node, end_node_exclusive):
            node_start, node_end = _parse_numa_physical_core_range(node)
            node_cores = node_end - node_start + 1
            node_max = per_node_max_instances.get(node, 0)
            print(
                f"NUMA node {node} cores: {node_start}-{node_end} "
                + f"(count={node_cores}, max_instances={node_max})"
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    server_args: List[str] = []
    server_env = config_env_vars
    server_host = config_host or "127.0.0.1"
    server_args.extend(config_server_cmd_args)

    if args.dry_run:
        sample_cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            *server_args,
            "--host",
            server_host,
            "--port",
            str(start_port),
        ]
        print("Command:", " ".join(sample_cmd))
        if args.print_env and server_env:
            print("Environment:")
            for k, v in server_env.items():
                print(f"  {k}={v}")
        return 0

    url_list = _launch_servers(
        start_node=start_node,
        end_node_exclusive=end_node_exclusive,
        total_instances=total_instances,
        cores_per_inst=cores_per_inst,
        start_port=start_port,
        output_dir=output_dir,
        server_args=server_args,
        server_host=server_host,
        server_env=server_env,
    )

    healthy_url_list = _health_check_servers(url_list)
    num_launched_servers = len(healthy_url_list)

    if not healthy_url_list:
        print("No reachable servers found. Exiting.")
        return 3

    healthy_url_list_str = " ".join(healthy_url_list)
    os.environ["HEALTHY_URL_LIST_STR"] = healthy_url_list_str

    print(f"Request producer started with URLs: {' '.join(healthy_url_list)}")
    print(f"Logs can be found in {output_dir}/run.log")
    print("To stop the request producer, use: killall -9 python3")
    print("To stop the servers, use: killall -9 python3")
    print("To check the status of the servers, use: ps aux | grep sglang.launch_server")
    print("To check the logs of the servers, use: tail -f server_*.log")
    print("To check the logs of the request producer, use: tail -f request_producer.log")
    print(
        "To check the status of the launched servers, use: curl -s -o /dev/null -w '%{http_code}' "
        + " ".join(healthy_url_list)
    )
    print(f"Healthy servers: {num_launched_servers}")

    router_log = output_dir / "router.log"
    with open(router_log, "wb"):
        pass
    router_cmd = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--worker-urls",
        *healthy_url_list,
    ]

    max_running_raw = _extract_flag_value(
        config_server_cmd_args,
        "--max-running-requests",
    )
    if max_running_raw is not None:
        try:
            max_running = int(max_running_raw)
            max_concurrent = max_running * len(healthy_url_list)
            router_cmd.extend(["--max-concurrent-requests", str(max_concurrent)])
        except ValueError:
            print(
                f"Invalid max-running-requests value: {max_running_raw}. "
                "Skipping router --max-concurrent-requests.",
                file=sys.stderr,
            )

    router_cmd.extend(config_router_cmd_args)

    print("Launching router:", " ".join(shlex.quote(x) for x in router_cmd))
    if args.detach_router:
        router_fh = open(router_log, "wb", buffering=0)
        router_proc = subprocess.Popen(
            router_cmd,
            stdout=router_fh,
            stderr=subprocess.STDOUT,
        )
        print(f"Router started in background with PID {router_proc.pid}")
        toc = int(time.time())
        elapsed_time = toc - tic
        print(f"Total time taken to launch servers: {elapsed_time} seconds")
        return 0

    with open(router_log, "wb", buffering=0) as router_fh:
        router_ret = subprocess.run(
            router_cmd,
            stdout=router_fh,
            stderr=subprocess.STDOUT,
        ).returncode

    toc = int(time.time())
    elapsed_time = toc - tic
    print(f"Total time taken to launch servers: {elapsed_time} seconds")
    return router_ret



if __name__ == "__main__":
    sys.exit(main())
