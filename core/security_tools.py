"""Tool implementations for the Security Agent tab.

Every tool that touches a real target goes through ScopeManager.is_in_scope()
before it's even offered to the user for approval (see core/tool_agent.py) —
this module only implements *what a tool does once approved*, not scope
enforcement itself.

None of this is sandboxed beyond process-level timeouts. The safety model is:
explicit scope allowlist + human approval of the exact command/code before
every single execution — not automated containment. Treat it accordingly.
"""

import ipaddress
import json
import logging
import os
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger("ai_agent_loader.security_tools")

_MAX_OUTPUT_CHARS = 8000


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: Optional[str] = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema, OpenAI function-calling style
    func: Callable[..., ToolResult]
    target_field: Optional[str] = None  # which param holds the scope-checked target
    requires_binary: Optional[str] = None  # external executable this needs, if any


def _truncate(text: str) -> str:
    if len(text) > _MAX_OUTPUT_CHARS:
        return text[:_MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text)} chars total]"
    return text


def extract_host(value: str) -> str:
    """Pulls a bare host/IP out of either a URL or a plain host string.

    CIDR-shaped values (e.g. "10.0.0.0/24") are returned unchanged rather
    than truncated to the network address — scope must check the *whole*
    requested range, otherwise an entry scoped to a single IP could be
    abused by appending a wide "/0"-style suffix that gets silently
    stripped before the scope check but still reaches the actual scanner.
    """
    value = value.strip()
    if "://" in value:
        parsed = urlparse(value)
        return parsed.hostname or value
    if "/" in value:
        try:
            ipaddress.ip_network(value, strict=False)
            return value
        except ValueError:
            pass
    return value.split("/")[0].split(":")[0]


# ---------------------------------------------------------------------------
# Passive / low-impact recon
# ---------------------------------------------------------------------------

def http_get(url: str, headers: Optional[Dict[str, str]] = None) -> ToolResult:
    try:
        resp = requests.get(url, headers=headers or {}, timeout=15, allow_redirects=True)
        body = resp.text[:4000]
        out = (
            f"GET {url}\n"
            f"Status: {resp.status_code}\n"
            f"Final URL: {resp.url}\n"
            f"Headers: {json.dumps(dict(resp.headers), indent=2)}\n\n"
            f"Body (first 4000 chars):\n{body}"
        )
        return ToolResult(success=True, output=_truncate(out))
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def dns_lookup(domain: str, record_type: str = "A") -> ToolResult:
    try:
        import dns.resolver

        record_type = (record_type or "A").upper()
        answers = dns.resolver.resolve(domain, record_type)
        records = [str(r) for r in answers]
        return ToolResult(success=True, output=f"{record_type} records for {domain}:\n" + "\n".join(records))
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def ssl_check(host: str, port: int = 443) -> ToolResult:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as tls:
                cert = tls.getpeercert()
                cipher = tls.cipher()
                version = tls.version()
        out = (
            f"TLS check for {host}:{port}\n"
            f"Protocol: {version}\n"
            f"Cipher: {cipher}\n"
            f"Subject: {cert.get('subject')}\n"
            f"Issuer: {cert.get('issuer')}\n"
            f"Not Before: {cert.get('notBefore')}\n"
            f"Not After: {cert.get('notAfter')}\n"
            f"Subject Alt Names: {cert.get('subjectAltName')}\n"
        )
        return ToolResult(success=True, output=out)
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def subdomain_enum(domain: str) -> ToolResult:
    """Passive subdomain enumeration via certificate-transparency logs (crt.sh)."""
    try:
        resp = requests.get(f"https://crt.sh/?q=%25.{domain}&output=json", timeout=30)
        resp.raise_for_status()
        entries = resp.json()
        names = set()
        for entry in entries:
            for name in str(entry.get("name_value", "")).split("\n"):
                name = name.strip().lstrip("*.")
                if name and domain in name:
                    names.add(name)
        sorted_names = sorted(names)[:200]
        out = f"Found {len(names)} unique subdomains for {domain} (showing up to 200):\n" + "\n".join(sorted_names)
        return ToolResult(success=True, output=_truncate(out))
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def shodan_search(query: str) -> ToolResult:
    api_key = os.getenv("SHODAN_API_KEY")
    if not api_key:
        return ToolResult(
            success=False,
            error="No Shodan API key configured. Set SHODAN_API_KEY in your .env file (get one from "
            "your own account at shodan.io — this tool never sources or shares credentials on your behalf).",
        )
    try:
        resp = requests.get(
            "https://api.shodan.io/shodan/host/search",
            params={"key": api_key, "query": query},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("matches", [])
        lines = [f"Total results: {data.get('total', len(matches))}"]
        for m in matches[:20]:
            lines.append(
                f"- {m.get('ip_str')}:{m.get('port')} | {m.get('org', '?')} | "
                f"{(m.get('data') or '').splitlines()[0][:120]}"
            )
        return ToolResult(success=True, output=_truncate("\n".join(lines)))
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# ---------------------------------------------------------------------------
# Active scanning
# ---------------------------------------------------------------------------

def host_discovery(cidr: str, timeout_ms: int = 800) -> ToolResult:
    """Ping-sweeps a CIDR range (or single host) to find which addresses
    respond, so a whole subnet doesn't have to be port-scanned host by host
    just to find out what's actually up. Shells out to the OS 'ping' binary
    per address (present on Windows/Linux/macOS by default, no extra
    installs or elevated privileges needed) rather than raw ICMP sockets."""
    import concurrent.futures

    try:
        network = ipaddress.ip_network(cidr, strict=False)
    except ValueError as e:
        return ToolResult(success=False, error=f"Invalid target '{cidr}': {e}")

    hosts = [network.network_address] if network.num_addresses <= 2 else list(network.hosts())
    cap = 1024
    truncated = len(hosts) > cap
    hosts = hosts[:cap]

    timeout_s = max(0.1, min(int(timeout_ms or 800), 5000) / 1000)

    def _ping(ip) -> bool:
        ip_str = str(ip)
        if sys.platform == "win32":
            args = ["ping", "-n", "1", "-w", str(int(timeout_s * 1000)), ip_str]
        else:
            args = ["ping", "-c", "1", "-W", str(max(1, round(timeout_s))), ip_str]
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s + 2)
            return proc.returncode == 0
        except Exception:
            return False

    alive: List[str] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            futures = {pool.submit(_ping, ip): ip for ip in hosts}
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    alive.append(str(futures[future]))
    except Exception as e:
        return ToolResult(success=False, error=str(e))

    alive.sort(key=lambda s: ipaddress.ip_address(s))
    lines = [
        f"Host discovery for {cidr} — {len(hosts)} address(es) checked"
        + (" (capped at 1024)" if truncated else "")
        + f", {len(alive)} responded to ping:"
    ]
    lines.extend(f"  {ip}" for ip in alive)
    if not alive:
        lines.append("  (none responded — hosts may still be up but blocking ICMP; try port_scan on specific candidates)")
    return ToolResult(success=True, output=_truncate("\n".join(lines)))


def _parse_port_spec(ports: str, cap: int = 4096) -> List[int]:
    result: List[int] = []
    for chunk in (ports or "1-1024").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(chunk))
    # De-dupe, sort, and hard-cap so one call can't turn into a multi-hour
    # (or accidentally DoS-shaped) sweep.
    result = sorted(set(p for p in result if 1 <= p <= 65535))
    return result[:cap]


def port_scan(target: str, ports: str = "1-1024", timeout_ms: int = 500) -> ToolResult:
    import concurrent.futures

    port_list = _parse_port_spec(ports)
    timeout_s = max(0.05, min(timeout_ms, 5000) / 1000)
    open_ports: List[int] = []

    def _check(port: int) -> Optional[int]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout_s)
                if s.connect_ex((target, port)) == 0:
                    return port
        except Exception:
            pass
        return None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as pool:
            for result in pool.map(_check, port_list):
                if result:
                    open_ports.append(result)
        open_ports.sort()
        out = (
            f"TCP connect scan of {target} ({len(port_list)} ports checked)\n"
            f"Open ports: {open_ports if open_ports else 'none found'}"
        )
        return ToolResult(success=True, output=out)
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _run_external_tool(binary: str, args: List[str], timeout: int = 300) -> ToolResult:
    path = shutil.which(binary)
    if not path:
        return ToolResult(
            success=False,
            error=f"'{binary}' is not installed (or not on PATH). Install it and try again — "
            f"this app does not install scanning tools on its own.",
        )
    try:
        proc = subprocess.run(
            [path] + args, capture_output=True, text=True, timeout=timeout
        )
        combined = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return ToolResult(success=proc.returncode == 0, output=_truncate(combined), error=None if proc.returncode == 0 else f"exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error=f"'{binary}' timed out after {timeout}s")
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def nmap_scan(target: str, args: str = "-sV -T4") -> ToolResult:
    try:
        extra = shlex.split(args or "")
    except ValueError as e:
        return ToolResult(success=False, error=f"Could not parse args: {e}")
    return _run_external_tool("nmap", extra + [target], timeout=600)


def nikto_scan(target: str) -> ToolResult:
    return _run_external_tool("nikto", ["-h", target], timeout=600)


def nuclei_scan(target: str, templates: Optional[str] = None) -> ToolResult:
    args = ["-u", target, "-silent"]
    if templates:
        args += ["-t", templates]
    return _run_external_tool("nuclei", args, timeout=600)


# ---------------------------------------------------------------------------
# LLM-authored exploit / PoC execution
# ---------------------------------------------------------------------------

def exploit_runner(code: str, language: str = "python", timeout: int = 60) -> ToolResult:
    """Executes LLM-authored code as-is, after human approval.

    There is no sandboxing beyond a process timeout — it runs with the same
    privileges as this app, on this machine. The approval step (which shows
    the *full* code before this ever executes) is the actual safety
    boundary, not this function.
    """
    language = (language or "python").lower()
    timeout = max(1, min(int(timeout or 60), 300))

    suffix = {"python": ".py", "bash": ".sh", "sh": ".sh"}.get(language, ".py")
    interpreter = {
        "python": [sys.executable],
        "bash": ["bash"],
        "sh": ["sh"],
    }.get(language, [sys.executable])

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        proc = subprocess.run(
            interpreter + [tmp_path], capture_output=True, text=True, timeout=timeout
        )
        combined = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return ToolResult(
            success=proc.returncode == 0,
            output=_truncate(combined),
            error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error=f"Execution timed out after {timeout}s")
    except Exception as e:
        return ToolResult(success=False, error=str(e))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _string_param(desc: str) -> Dict[str, Any]:
    return {"type": "string", "description": desc}


TOOLS: Dict[str, ToolSpec] = {}


def _register(spec: ToolSpec) -> None:
    TOOLS[spec.name] = spec


_register(ToolSpec(
    name="http_get",
    description="Make an HTTP GET request and return status, headers, and body.",
    parameters={
        "type": "object",
        "properties": {
            "url": _string_param("Full URL to request, e.g. https://example.com/"),
        },
        "required": ["url"],
    },
    func=lambda url, headers=None: http_get(url, headers),
    target_field="url",
))

_register(ToolSpec(
    name="dns_lookup",
    description="Resolve DNS records for a domain.",
    parameters={
        "type": "object",
        "properties": {
            "domain": _string_param("Domain name to resolve, e.g. example.com"),
            "record_type": _string_param("DNS record type: A, AAAA, MX, TXT, NS, CNAME, SOA. Default A."),
        },
        "required": ["domain"],
    },
    func=dns_lookup,
    target_field="domain",
))

_register(ToolSpec(
    name="ssl_check",
    description="Connect via TLS and report the certificate details (subject, issuer, expiry, SANs, cipher).",
    parameters={
        "type": "object",
        "properties": {
            "host": _string_param("Hostname to connect to"),
            "port": {"type": "integer", "description": "Port, default 443"},
        },
        "required": ["host"],
    },
    func=ssl_check,
    target_field="host",
))

_register(ToolSpec(
    name="subdomain_enum",
    description="Passively enumerate subdomains for a domain using certificate-transparency logs (crt.sh).",
    parameters={
        "type": "object",
        "properties": {
            "domain": _string_param("Root domain, e.g. example.com"),
        },
        "required": ["domain"],
    },
    func=subdomain_enum,
    target_field="domain",
))

_register(ToolSpec(
    name="shodan_search",
    description="Search Shodan's database (requires SHODAN_API_KEY to be set in .env). Not target-scoped, "
    "since it queries Shodan's own data rather than contacting a host directly.",
    parameters={
        "type": "object",
        "properties": {
            "query": _string_param("Shodan search query, e.g. 'apache country:US'"),
        },
        "required": ["query"],
    },
    func=shodan_search,
    target_field=None,
))

_register(ToolSpec(
    name="host_discovery",
    description="Ping-sweep a CIDR range (or single host) to find which addresses are currently up. "
    "Run this before port_scan/nmap on a whole subnet so you're not scanning dead addresses.",
    parameters={
        "type": "object",
        "properties": {
            "cidr": _string_param("CIDR range or single host/IP, e.g. '10.0.0.0/24' or '10.0.0.5'"),
            "timeout_ms": {"type": "integer", "description": "Per-host ping timeout in ms, default 800"},
        },
        "required": ["cidr"],
    },
    func=host_discovery,
    target_field="cidr",
))

_register(ToolSpec(
    name="port_scan",
    description="Pure-Python TCP connect scan (no external tool needed). Capped at 4096 ports per call.",
    parameters={
        "type": "object",
        "properties": {
            "target": _string_param("Hostname or IP to scan"),
            "ports": _string_param("Port spec, e.g. '1-1024' or '22,80,443'. Default 1-1024."),
            "timeout_ms": {"type": "integer", "description": "Per-port connect timeout in ms, default 500"},
        },
        "required": ["target"],
    },
    func=port_scan,
    target_field="target",
))

_register(ToolSpec(
    name="nmap",
    description="Run nmap against a target (requires nmap installed on this machine).",
    parameters={
        "type": "object",
        "properties": {
            "target": _string_param("Hostname, IP, or CIDR to scan"),
            "args": _string_param("nmap arguments, e.g. '-sV -T4 -p-'. Default '-sV -T4'."),
        },
        "required": ["target"],
    },
    func=nmap_scan,
    target_field="target",
    requires_binary="nmap",
))

_register(ToolSpec(
    name="nikto",
    description="Run a nikto web-server scan against a target (requires nikto installed).",
    parameters={
        "type": "object",
        "properties": {
            "target": _string_param("URL or host to scan, e.g. https://example.com"),
        },
        "required": ["target"],
    },
    func=nikto_scan,
    target_field="target",
    requires_binary="nikto",
))

_register(ToolSpec(
    name="nuclei",
    description="Run nuclei vulnerability-scanning templates against a target (requires nuclei installed).",
    parameters={
        "type": "object",
        "properties": {
            "target": _string_param("URL or host to scan"),
            "templates": _string_param("Optional template path/tag filter, e.g. 'cves/' or 'http/exposures'"),
        },
        "required": ["target"],
    },
    func=nuclei_scan,
    target_field="target",
    requires_binary="nuclei",
))

_register(ToolSpec(
    name="exploit_runner",
    description="Execute a script YOU (the model) write, e.g. a PoC exercising a specific finding. "
    "There is no sandboxing beyond a timeout — the human approving this call will see the full code "
    "before it runs. Include a comment at the top of the code naming the exact target host, and only "
    "ever target hosts already confirmed to be in scope.",
    parameters={
        "type": "object",
        "properties": {
            "code": _string_param("Full source code to execute"),
            "language": _string_param("python, bash, or sh. Default python."),
            "timeout": {"type": "integer", "description": "Max seconds to allow, default 60, capped at 300"},
        },
        "required": ["code"],
    },
    func=exploit_runner,
    target_field=None,
))


def get_tool(name: str) -> Optional[ToolSpec]:
    return TOOLS.get(name)


def list_tools() -> List[ToolSpec]:
    return list(TOOLS.values())


def openai_tool_schemas() -> List[Dict[str, Any]]:
    """Tool definitions in OpenAI function-calling `tools=[...]` format."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in TOOLS.values()
    ]
