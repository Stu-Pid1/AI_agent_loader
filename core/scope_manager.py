import ipaddress
import json
import logging
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("ai_agent_loader.scope_manager")

SCOPE_FILE = Path(__file__).parent.parent / "agent_scope.json"


@dataclass
class ScopeEntry:
    target: str  # hostname, IP, or CIDR (e.g. "example.com", "10.0.0.5", "10.0.0.0/24")
    note: str = ""
    added_at: str = ""


class ScopeManager:
    """Hard-enforced allowlist of targets the security-agent tools may act on.

    This is the one safety mechanism every tool in core/security_tools.py
    goes through — nothing runs against a host that isn't explicitly listed
    here first, same as a signed scope document on a real engagement. It is
    not a suggestion the agent can talk its way around; is_in_scope() is a
    hard gate checked before any tool call is even offered for approval.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else SCOPE_FILE
        self._entries: List[ScopeEntry] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = []
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._entries = [ScopeEntry(**e) for e in data]
        except Exception as e:
            logger.warning(f"Could not read scope file {self._path}: {e}")
            self._entries = []

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([asdict(e) for e in self._entries], indent=2), encoding="utf-8"
        )

    def list_entries(self) -> List[ScopeEntry]:
        return list(self._entries)

    def add(self, target: str, note: str = "") -> None:
        target = target.strip()
        if not target:
            return
        if any(e.target == target for e in self._entries):
            return
        self._entries.append(
            ScopeEntry(target=target, note=note, added_at=datetime.now(timezone.utc).isoformat())
        )
        self._save()

    def remove(self, target: str) -> bool:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.target != target]
        if len(self._entries) != before:
            self._save()
            return True
        return False

    def clear(self) -> None:
        self._entries = []
        self._save()

    def is_in_scope(self, target: str) -> bool:
        """True if `target` (hostname, IP, or CIDR) matches a scope entry.

        Hostnames match exactly (or as a subdomain of a scoped domain, e.g.
        'api.example.com' is in scope if 'example.com' is listed). IPs match
        exact-IP entries or any CIDR entry that contains them. Hostnames are
        also resolved and checked against CIDR entries, since "is this IP
        authorized" is usually what actually matters for scanning tools.

        A CIDR-shaped target (e.g. a ping-sweep or nmap range) is only in
        scope if the *entire* requested range is contained within a scoped
        CIDR/IP entry — not just its network address — so a tool can't be
        pointed at a range wider than what was actually authorized.
        """
        target = target.strip().rstrip(".")
        if not target:
            return False

        if "/" in target:
            return self._cidr_target_in_scope(target)

        for entry in self._entries:
            scoped = entry.target.strip().rstrip(".")
            if not scoped:
                continue

            if "/" in scoped:
                try:
                    network = ipaddress.ip_network(scoped, strict=False)
                except ValueError:
                    continue
                if self._ip_in_network(target, network):
                    return True
                continue

            if self._looks_like_ip(scoped):
                if target == scoped:
                    return True
                continue

            # Hostname / domain entry: exact match or subdomain match.
            if target == scoped or target.endswith("." + scoped):
                return True

        return False

    def _cidr_target_in_scope(self, target: str) -> bool:
        try:
            target_network = ipaddress.ip_network(target, strict=False)
        except ValueError:
            return False

        for entry in self._entries:
            scoped = entry.target.strip().rstrip(".")
            if not scoped:
                continue
            try:
                if "/" in scoped:
                    scoped_network = ipaddress.ip_network(scoped, strict=False)
                elif self._looks_like_ip(scoped):
                    bits = 32 if ipaddress.ip_address(scoped).version == 4 else 128
                    scoped_network = ipaddress.ip_network(f"{scoped}/{bits}", strict=False)
                else:
                    continue  # hostnames can't bound a CIDR target
            except ValueError:
                continue

            if target_network.version != scoped_network.version:
                continue
            if target_network == scoped_network or target_network.subnet_of(scoped_network):
                return True

        return False

    @staticmethod
    def _looks_like_ip(value: str) -> bool:
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    @classmethod
    def _ip_in_network(cls, target: str, network) -> bool:
        candidates = [target] if cls._looks_like_ip(target) else cls._resolve_all(target)
        for candidate in candidates:
            try:
                if ipaddress.ip_address(candidate) in network:
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _resolve_all(hostname: str) -> List[str]:
        try:
            infos = socket.getaddrinfo(hostname, None)
            return list({info[4][0] for info in infos})
        except Exception:
            return []
