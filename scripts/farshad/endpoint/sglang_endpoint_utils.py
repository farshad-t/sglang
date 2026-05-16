import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise

def _normalize_host(host: str) -> str:
    if host.startswith("http://"):
        return host[len("http://") :]
    if host.startswith("https://"):
        return host[len("https://") :]
    return host


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate_sglang_config_schema(config: Dict[str, Any]) -> None:
    def require_keys(obj: Dict[str, Any], keys: List[str], ctx: str) -> None:
        for k in keys:
            if k not in obj:
                raise ValueError(f"Missing required key '{k}' in {ctx}")

    def ensure_type(value: Any, expected: str, ctx: str) -> None:
        actual = _type_name(value)
        if expected != actual:
            raise ValueError(f"Expected type '{expected}' for {ctx}, got '{actual}'")

    require_keys(config, ["name", "platform", "endpoint"], "config")
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

        has_server_cmd_args = "server_cmd_args" in ep and ep["server_cmd_args"] is not None
        has_cmd_args = "cmd_args" in ep and ep["cmd_args"] is not None
        if not has_server_cmd_args and not has_cmd_args:
            raise ValueError(f"Missing required key 'server_cmd_args' or 'cmd_args' in {ctx}")

        if has_server_cmd_args:
            ensure_type(ep["server_cmd_args"], "array", f"{ctx}.server_cmd_args")
        if has_cmd_args:
            ensure_type(ep["cmd_args"], "array", f"{ctx}.cmd_args")

        if "envs" in ep and ep["envs"] is not None:
            ensure_type(ep["envs"], "array", f"{ctx}.envs")
            for j, env in enumerate(ep["envs"]):
                ectx = f"{ctx}.envs[{j}]"
                ensure_type(env, "object", ectx)
                require_keys(env, ["name", "value"], ectx)
                ensure_type(env["name"], "string", f"{ectx}.name")

        if has_server_cmd_args:
            for j, arg in enumerate(ep["server_cmd_args"]):
                actx = f"{ctx}.server_cmd_args[{j}]"
                ensure_type(arg, "object", actx)
                require_keys(arg, ["name"], actx)
                ensure_type(arg["name"], "string", f"{actx}.name")

        if has_cmd_args:
            for j, arg in enumerate(ep["cmd_args"]):
                actx = f"{ctx}.cmd_args[{j}]"
                ensure_type(arg, "object", actx)
                require_keys(arg, ["name"], actx)
                ensure_type(arg["name"], "string", f"{actx}.name")

        if "router_cmd_args" in ep and ep["router_cmd_args"] is not None:
            ensure_type(ep["router_cmd_args"], "array", f"{ctx}.router_cmd_args")
            for j, arg in enumerate(ep["router_cmd_args"]):
                actx = f"{ctx}.router_cmd_args[{j}]"
                ensure_type(arg, "object", actx)
                require_keys(arg, ["name"], actx)
                ensure_type(arg["name"], "string", f"{actx}.name")


def _select_endpoint(
    config: Dict[str, Any], endpoint_name: Optional[str]
) -> Optional[Dict[str, Any]]:
    endpoints = config.get("endpoint", []) or []
    if endpoint_name:
        for ep in endpoints:
            if str(ep.get("name", "")) == endpoint_name:
                return ep
    if endpoints:
        return endpoints[0]
    return None


def _collect_envs(endpoint_cfg: Dict[str, Any]) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    for env in endpoint_cfg.get("envs", []) or []:
        name = str(env.get("name", "")).strip()
        if not name:
            continue
        env_vars[name] = str(env.get("value"))
    return env_vars


def _collect_server_cmd_args(
    endpoint_cfg: Dict[str, Any],
) -> Tuple[List[str], Optional[str], List[str]]:
    cmd_args: List[str] = []
    host_override: Optional[str] = None
    seen_names: List[str] = []

    cmd_args_key = "server_cmd_args" if "server_cmd_args" in endpoint_cfg else "cmd_args"
    for arg in endpoint_cfg.get(cmd_args_key, []) or []:
        name = str(arg.get("name", "")).strip()
        if not name:
            continue

        has_value = "value" in arg
        value = arg.get("value")

        if name == "host":
            if has_value and value is not None:
                host_override = _normalize_host(str(value))
            continue
        if name == "port":
            continue

        seen_names.append(name)
        flag = name if name.startswith("-") else f"--{name}"

        if not has_value:
            cmd_args.append(flag)
            continue

        if isinstance(value, bool):
            if _is_truthy(value):
                cmd_args.append(flag)
            continue

        if value is None:
            cmd_args.append(flag)
            continue

        cmd_args.extend([flag, str(value)])

    return cmd_args, host_override, seen_names


def _collect_router_cmd_args(
    endpoint_cfg: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    cmd_args: List[str] = []
    seen_names: List[str] = []

    for arg in endpoint_cfg.get("router_cmd_args", []) or []:
        name = str(arg.get("name", "")).strip()
        if not name:
            continue

        has_value = "value" in arg
        value = arg.get("value")

        normalized_name = name.lstrip("-")
        seen_names.append(normalized_name)

        flag = name if name.startswith("-") else f"--{name}"

        if not has_value:
            cmd_args.append(flag)
            continue

        if normalized_name == "host" and value is not None:
            value = _normalize_host(str(value))

        if isinstance(value, bool):
            if _is_truthy(value):
                cmd_args.append(flag)
            continue

        if value is None:
            cmd_args.append(flag)
            continue

        cmd_args.extend([flag, str(value)])

    return cmd_args, seen_names


def load_sglang_endpoint_config(
    config_path: str, endpoint_name: Optional[str]
) -> Tuple[Dict[str, str], List[str], Optional[str], List[str], List[str], List[str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_sglang_config_schema(config)

    selected = _select_endpoint(config, endpoint_name)
    if selected is None:
        raise KeyError("No endpoint entries found in config")

    env_vars = _collect_envs(selected)
    cmd_args, host_override, seen_names = _collect_server_cmd_args(selected)
    router_cmd_args, router_seen_names = _collect_router_cmd_args(selected)
    return env_vars, cmd_args, host_override, seen_names, router_cmd_args, router_seen_names
