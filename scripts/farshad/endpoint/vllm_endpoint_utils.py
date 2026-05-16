import os
import sys
import socket
import errno
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Basic schema (code-enforced) for config validation
# ----------------------------------------------------------------------------
SchemaType = Dict[str, Any]

CONFIG_SCHEMA: SchemaType = {
    "type": "object",
    "required": ["name", "platform", "endpoint"],
    "properties": {
        "name": {"type": "string"},
        "platform": {"type": "string"},
        "endpoint": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "envs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "value"],
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": ["string", "number", "boolean"]},
                            },
                        },
                    },
                    "cmd_args": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name"],  # value optional for presence-only flags
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": ["string", "number", "boolean"]},
                            },
                        },
                    },
                    "server_cmd_args": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name"],  # value optional for presence-only flags
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": ["string", "number", "boolean"]},
                            },
                        },
                    },
                },
            },
        },
    },
}


def _type_name(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def validate_config_schema(config: Dict[str, Any]) -> None:
    """Validate config against CONFIG_SCHEMA.
    Lightweight validator to avoid external dependencies.
    Raises ValueError on validation failure.
    """
    def require_keys(obj: Dict[str, Any], keys: List[str], ctx: str) -> None:
        for k in keys:
            if k not in obj:
                raise ValueError(f"Missing required key '{k}' in {ctx}")

    def ensure_type(value: Any, expected: str, ctx: str) -> None:
        actual = _type_name(value)
        if expected != actual:
            raise ValueError(f"Expected type '{expected}' for {ctx}, got '{actual}'")

    # Top-level
    require_keys(config, CONFIG_SCHEMA["required"], "config")
    ensure_type(config["name"], "string", "config.name")
    ensure_type(config["platform"], "string", "config.platform")
    ensure_type(config["endpoint"], "array", "config.endpoint")

    endpoints = config["endpoint"]
    if not endpoints:
        raise ValueError("config.endpoint must have at least one item")

    for i, ep in enumerate(endpoints):
        ctx = f"config.endpoint[{i}]"
        ensure_type(ep, "object", ctx)
        require_keys(ep, ["name"], ctx)
        ensure_type(ep["name"], "string", f"{ctx}.name")

        has_cmd_args = "cmd_args" in ep and ep["cmd_args"] is not None
        has_server_cmd_args = "server_cmd_args" in ep and ep["server_cmd_args"] is not None
        if not has_cmd_args and not has_server_cmd_args:
            raise ValueError(f"Missing required key 'cmd_args' or 'server_cmd_args' in {ctx}")

        if has_cmd_args:
            ensure_type(ep["cmd_args"], "array", f"{ctx}.cmd_args")
        if has_server_cmd_args:
            ensure_type(ep["server_cmd_args"], "array", f"{ctx}.server_cmd_args")

        if "envs" in ep and ep["envs"] is not None:
            ensure_type(ep["envs"], "array", f"{ctx}.envs")
            for j, env in enumerate(ep["envs"]):
                ectx = f"{ctx}.envs[{j}]"
                ensure_type(env, "object", ectx)
                require_keys(env, ["name", "value"], ectx)
                ensure_type(env["name"], "string", f"{ectx}.name")

        if has_cmd_args:
            for j, arg in enumerate(ep["cmd_args"]):
                actx = f"{ctx}.cmd_args[{j}]"
                ensure_type(arg, "object", actx)
                require_keys(arg, ["name"], actx)
                ensure_type(arg["name"], "string", f"{actx}.name")

        if has_server_cmd_args:
            for j, arg in enumerate(ep["server_cmd_args"]):
                actx = f"{ctx}.server_cmd_args[{j}]"
                ensure_type(arg, "object", actx)
                require_keys(arg, ["name"], actx)
                ensure_type(arg["name"], "string", f"{actx}.name")



# ----------------------------------------------------------------------------
# Mapping and command construction
# ----------------------------------------------------------------------------

# Boolean flags that should be passed as presence-only when True or when value missing.
BOOLEAN_PRESENCE_FLAGS = {
    "async-scheduling": "async-scheduling",
    "enable-prefix-caching": "enable-prefix-caching",
    "disable-log-requests": "disable-log-requests",
    "no-enable-prefix-caching": "no-enable-prefix-caching",
    "no-enable-expert-parallel": "no-enable-expert-parallel",
    "trust-request-chat-template": "trust-request-chat-template",
}

# Keys that are positional or special-handled
SPECIAL_KEYS = {"model", "host", "port"}


def normalize_host(host: str) -> str:
    if host.startswith("http://"):
        return host[len("http://") :]
    if host.startswith("https://"):
        return host[len("https://") :]
    return host


# Map various user-provided keys to canonical vLLM flag names
CANONICAL_FLAG_KEYS: Dict[str, str] = {
    # Tensor parallel aliases
    "tensor-parallel": "tensor-parallel-size",
    "tensor_parallel": "tensor-parallel-size",
    "tensor_parallel_size": "tensor-parallel-size",
    # Pipeline parallel aliases
    "pipeline-parallel": "pipeline-parallel-size",
    "pipeline_parallel": "pipeline-parallel-size",
    "pipeline_parallel_size": "pipeline-parallel-size",
}


def normalize_flag_key(key: str) -> str:
    return CANONICAL_FLAG_KEYS.get(key, key)


def build_vllm_command(endpoint_cfg: Dict[str, Any], top_level_devices: Optional[str] = None) -> Tuple[List[str], Dict[str, str]]:
    """Build the vllm serve command and environment variables.

    Returns (cmd_list, env_vars).
    """
    # Base command
    cmd: List[str] = ["vllm", "serve"]

    # Environment variables (endpoint-specific). Do NOT handle ZE_AFFINITY_MASK here.
    env_vars: Dict[str, str] = {}
    for env in endpoint_cfg.get("envs", []) or []:
        name = str(env.get("name", "")).strip()
        value = env.get("value")
        if not name:
            continue
        if name == "ZE_AFFINITY_MASK":
            # Endpoint-agnostic; ignore if provided under endpoint envs.
            continue
        env_vars[name] = str(value)

    # Devices / ZE_AFFINITY_MASK handling (endpoint-agnostic)
    devices_mask = extract_devices_mask(endpoint_cfg, top_level_devices)
    effective_mask = (os.environ.get("ZE_AFFINITY_MASK") or devices_mask 
                      if endpoint_cfg.get("platform", "").lower() == "xpu" else
                       os.environ.get("CPU_VISIBLE_MEMORY_NODES") or devices_mask
                    )
    dp_count: Optional[int] = count_devices_in_mask(effective_mask) if effective_mask else None

    # Extract cmd_args
    model_path: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None

    def is_truthy(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(v, (int, float)):
            return bool(v)
        return False

    seen_keys: Dict[str, bool] = {}
    for arg in endpoint_cfg.get("cmd_args", []) or []:
        key = str(arg.get("name", "")).strip()
        norm_key = normalize_flag_key(key)
        has_value = "value" in arg
        value = arg.get("value")
        if not key:
            continue

        if key == "model":
            if not has_value:
                raise ValueError("'model' requires a value (path to model)")
            model_path = str(value)
        elif key == "host":
            if not has_value:
                raise ValueError("'host' requires a value")
            host = normalize_host(str(value))
        elif key == "port":
            if not has_value:
                raise ValueError("'port' requires a numeric value")
            try:
                port = int(value)
            except Exception:
                raise ValueError(f"Invalid port value: {value}")
        elif norm_key in BOOLEAN_PRESENCE_FLAGS:
            include_flag = True if not has_value else is_truthy(value)
            if include_flag:
                cmd.append(f"--{BOOLEAN_PRESENCE_FLAGS[norm_key]}")
        else:
            if not has_value:
                raise ValueError(f"Flag '{key}' requires a value")
            # Cap parallelism flags to available devices from ZE_AFFINITY_MASK
            if dp_count is not None and norm_key in ("tensor-parallel-size", "data-parallel-size", "api-server-count", "pipeline-parallel-size"):
                try:
                    int_val = int(value)
                except Exception:
                    raise ValueError(f"Flag '{key}' expects an integer value, got: {value}")
                eff_val = min(int_val, dp_count)
                if eff_val != int_val:
                    print(
                        f"Warning: flag '{norm_key}' requested {int_val} exceeds available devices {dp_count}; using {eff_val}.",
                        file=sys.stderr,
                    )
                cmd.extend([f"--{norm_key}", str(eff_val)])
            else:
                cmd.extend([f"--{norm_key}", str(value)])
        seen_keys[norm_key] = True

    # Positional model first (if provided)
    if model_path:
        cmd.insert(2, model_path)  # after 'vllm', 'serve'

    # Host/Port
    if host:
        cmd.extend(["--host", host])
    if port is not None:
        cmd.extend(["--port", str(port)])

    # No auto-insertion/override for parallel flags; only enforcement above

    return cmd, env_vars


# ----------------------------------------------------------------------------
# Utilities: extracting args, host/port, and path validation
# ----------------------------------------------------------------------------

def extract_arg_map(endpoint_cfg: Dict[str, Any]) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for arg in endpoint_cfg.get("cmd_args", []) or []:
        name = str(arg.get("name", "")).strip()
        if not name:
            continue
        if "value" in arg:
            m[name] = arg["value"]
        else:
            m[name] = None
    return m


def extract_devices_mask(endpoint_cfg: Dict[str, Any], top_level_devices: Optional[str] = None) -> Optional[str]:
    # Prefer top-level 'devices' if provided
    if top_level_devices is not None:
        s = str(top_level_devices).strip()
        if s:
            return s
    # Support either direct 'devices' field on endpoint or a cmd_arg named 'devices'
    if "devices" in endpoint_cfg:
        val = endpoint_cfg.get("devices")
        if val is not None:
            return str(val)
    # Fallback to cmd_args
    for arg in endpoint_cfg.get("cmd_args", []) or []:
        name = str(arg.get("name", "")).strip()
        if name == "devices":
            if "value" in arg:
                return str(arg.get("value"))
            # presence-only devices: treat as None (no mask)
            return None
    return None


def extract_host_port(endpoint_cfg: Dict[str, Any]) -> Tuple[str, int]:
    args = extract_arg_map(endpoint_cfg)
    host = args.get("host")
    port = args.get("port")
    host_str = str(host) if host is not None else "0.0.0.0"
    try:
        port_int = int(port) if port is not None else 8000
    except Exception:
        port_int = 8000
    # Normalize host the same way as used for flags
    host_str = normalize_host(host_str)
    return host_str, port_int


def _looks_like_path(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith(("/", "./", "../")):
        return True
    if os.sep in value or (os.altsep and os.altsep in value):
        return True
    if value.lower().startswith(("file://",)):
        return True
    # Heuristic for templates/files
    if any(value.lower().endswith(ext) for ext in (".jinja", ".json", ".safetensors", ".bin")):
        return True
    return False


def _resolve_path(base_dir: str, value: str) -> str:
    v = os.path.expandvars(os.path.expanduser(value))
    if os.path.isabs(v):
        return v
    return os.path.normpath(os.path.join(base_dir, v))


def validate_paths(endpoint_cfg: Dict[str, Any], base_dir: str) -> List[str]:
    """Validate path-like arguments exist. Returns a list of error strings.

    Heuristics:
    - Always check keys: 'model', 'chat-template' when they look like filesystem paths.
    - Also check any arg name containing: 'path', 'dir', 'file', 'template'.
    Paths are resolved relative to base_dir if not absolute, after env expansion.
    """
    errors: List[str] = []
    args = extract_arg_map(endpoint_cfg)

    def maybe_check(key: str) -> None:
        val = args.get(key)
        if val is None:
            return
        s = str(val)
        if not _looks_like_path(s):
            return
        resolved = _resolve_path(base_dir, s)
        if not os.path.exists(resolved):
            errors.append(f"Path for '{key}' does not exist: {resolved}")

    # Always check these when path-like
    for k in ("model", "chat-template"):
        maybe_check(k)

    # Heuristic keys
    for k in list(args.keys()):
        kl = k.lower()
        if any(t in kl for t in ("path", "dir", "file", "template")):
            maybe_check(k)

    return errors


def is_port_available(host: str, port: int) -> bool:
    """Return True if the TCP port appears available on the given host.

    Tries to bind a temporary socket; if EADDRINUSE, the port is in use.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Avoid lingering TIME_WAIT issues; we only probe binding capability.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            return False
        # For other errors (e.g., permission), conservatively treat as unavailable.
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def compute_probe_hosts(bind_host: str) -> List[str]:
    """Return a list of hosts to probe for readiness.

    If bind_host is a wildcard (e.g., 0.0.0.0 or ::), it cannot be used
    as a destination for HTTP requests. In that case, default to localhost
    addresses that typically route: 127.0.0.1 and 'localhost'.
    Otherwise, try the bind_host first, then localhost fallbacks.
    """
    h = (bind_host or "").strip()
    wildcards = {"0.0.0.0", "::", "0:0:0:0:0:0:0:0", ""}
    if h in wildcards:
        return ["127.0.0.1", "localhost"]
    return [h, "127.0.0.1", "localhost"]


def count_devices_in_mask(mask: Optional[str]) -> Optional[int]:
    if not mask:
        return None
    parts = [p.strip() for p in str(mask).split(",")]
    return len([p for p in parts if p]) or None
