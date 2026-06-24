#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AIOS — EtherSentinel : Direct MAC-Layer Network Fabric                    ║
║   Module  : aios_network.py                                                 ║
║                                                                             ║
║   "One wire. One MAC. One truth. Zero leakage."                             ║
║                                                                             ║
║   Layer Model (bottom-up):                                                  ║
║     L2  IEEE 802.3      EtherSentinel owns the wire via AF_PACKET          ║
║     L2.5 ARP RFC 826    Resolution of router IP → MAC only                 ║
║     L3  IPv4 RFC 791    Full codec, checksum, no kernel IP stack            ║
║     L4  UDP  RFC 768    Full codec, ephemeral-port managed                  ║
║     L7  DNS  RFC 1035   Stub resolver → custom server, never router        ║
║                                                                             ║
║   Policy Invariants (enforced at every send/receive path):                  ║
║     1. ALL outbound Ethernet frames are addressed to router MAC only        ║
║     2. ARP broadcast is the sole exception (router discovery only)          ║
║     3. DNS queries exit to custom_dns_ip:53 exclusively                     ║
║     4. No /etc/resolv.conf, no getaddrinfo(), no kernel routing table      ║
║     5. PacketFilter drops and counts every non-whitelisted frame            ║
║     6. custom_dns_ip ≠ router_ip is a compile-time invariant               ║
║                                                                             ║
║   Threading Model:                                                          ║
║     • PhysicalInterface: one daemon RX thread, one TX lock                 ║
║     • ARPCache: RLock + per-IP threading.Event for resolution wait          ║
║     • DNSStub: per-query threading.Event keyed on DNS query ID             ║
║                                                                             ║
║   AIOS Integration:                                                         ║
║     Every public operation is @agent_method decorated with full             ║
║     AgentTrace capture feeding the singleton AgentRegistry.                 ║
║                                                                             ║
║   Requires:                                                                 ║
║     Linux kernel ≥ 3.0, CAP_NET_RAW or UID 0                              ║
║     Python ≥ 3.9 (no third-party dependencies)                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import time
import struct
import socket
import fcntl
import hashlib
import random
import errno
import platform
import functools
import threading
import traceback
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
)
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag, auto
from collections import defaultdict, deque
from contextlib import contextmanager

# ── AIOS integration — import from kernel or fall back to minimal shims ──────
try:
    from aios_core import (
        agent_method,
        AgentPriority,
        AgentTrace,
        AgentContext,
    )
    _AIOS_INTEGRATED: bool = True
except ImportError:
    _AIOS_INTEGRATED = False

    class AgentPriority(IntEnum):  # type: ignore[no-redef]
        CRITICAL = 0
        HIGH     = 1
        NORMAL   = 2
        LOW      = 3

    def agent_method(                       # type: ignore[no-redef]
        name:        Optional[str]      = None,
        description: str                = "",
        parameters:  Optional[Dict]     = None,
        returns:     str                = "Any",
        priority:    Any                = None,
        owner:       str                = "network",
    ) -> Callable:
        """Passthrough shim when running outside the AIOS kernel."""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs.pop("_ctx", None)
                return fn(*args, **kwargs)
            return wrapper
        return decorator


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 0 — PROTOCOL CONSTANTS
# All numeric constants are derived from their respective RFC/IEEE standard.
# ════════════════════════════════════════════════════════════════════════════════

# ── EtherType values (IEEE 802.3 §3.2.6) ─────────────────────────────────────
ETH_P_IP:  int = 0x0800   # Internet Protocol version 4
ETH_P_ARP: int = 0x0806   # Address Resolution Protocol
ETH_P_ALL: int = 0x0003   # Capture every EtherType (AF_PACKET bind)

# ── Frame geometry (IEEE 802.3) ───────────────────────────────────────────────
ETH_HEADER_LEN:  int = 14   # dst(6) + src(6) + type(2)
ETH_MIN_PAYLOAD: int = 46   # minimum payload to meet 64-byte minimum frame
ETH_MAX_PAYLOAD: int = 1500 # maximum payload (MTU), no jumbo frames

# ── ARP (RFC 826) ─────────────────────────────────────────────────────────────
ARP_OP_REQUEST:   int = 1
ARP_OP_REPLY:     int = 2
ARP_HW_ETHERNET:  int = 1       # hardware type: Ethernet
ARP_PROTO_IPv4:   int = 0x0800  # protocol type: IPv4
ARP_PACKET_LEN:   int = 28      # fixed for Ethernet/IPv4

# ── IPv4 (RFC 791) ────────────────────────────────────────────────────────────
IP_HEADER_MIN_LEN: int = 20   # no options
IP_VERSION_IHL:    int = 0x45 # version=4, IHL=5 (20 bytes)
IP_FLAG_DF:        int = 0x4000  # Don't Fragment

# ── Transport protocols ───────────────────────────────────────────────────────
IPPROTO_ICMP: int = 1
IPPROTO_TCP:  int = 6
IPPROTO_UDP:  int = 17

# ── UDP (RFC 768) ─────────────────────────────────────────────────────────────
UDP_HEADER_LEN: int = 8
DNS_PORT:       int = 53

# ── DNS (RFC 1035) ────────────────────────────────────────────────────────────
DNS_TYPE_A:     int = 1
DNS_TYPE_NS:    int = 2
DNS_TYPE_CNAME: int = 5
DNS_TYPE_SOA:   int = 6
DNS_TYPE_MX:    int = 15
DNS_TYPE_AAAA:  int = 28
DNS_TYPE_ANY:   int = 255
DNS_CLASS_IN:   int = 1

DNS_FLAG_QR:    int = 0x8000  # 0=query, 1=response
DNS_FLAG_AA:    int = 0x0400  # authoritative answer
DNS_FLAG_TC:    int = 0x0200  # truncated message
DNS_FLAG_RD:    int = 0x0100  # recursion desired
DNS_FLAG_RA:    int = 0x0080  # recursion available

# ── Operational tuning ────────────────────────────────────────────────────────
ARP_CACHE_TTL_S:      float = 300.0   # ARP entry lifetime (seconds)
ARP_REQUEST_TIMEOUT:  float = 5.0    # wait for ARP reply
DNS_QUERY_TIMEOUT:    float = 5.0    # wait for DNS response
DNS_MAX_RETRIES:      int   = 3
EPHEMERAL_PORT_LO:    int   = 49152
EPHEMERAL_PORT_HI:    int   = 65535
RX_SOCKET_TIMEOUT:    float = 0.5    # non-blocking RX loop granularity

# ── Linux ioctl codes (from <linux/sockios.h>) ────────────────────────────────
SIOCGIFHWADDR: int = 0x8927   # get hardware address
SIOCGIFINDEX:  int = 0x8933   # get interface index
SIOCGIFADDR:   int = 0x8915   # get interface IP address

# ── Linux packet types (from <linux/if_packet.h> PACKET_*) ───────────────────
PACKET_HOST:      int = 0   # addressed to this host
PACKET_BROADCAST: int = 1   # link-layer broadcast
PACKET_MULTICAST: int = 2   # link-layer multicast group
PACKET_OTHERHOST: int = 3   # to other host (promiscuous mode)
PACKET_OUTGOING:  int = 4   # originated by this host

# ── Special MAC addresses ─────────────────────────────────────────────────────
_MAC_BROADCAST_BYTES: bytes = b'\xff\xff\xff\xff\xff\xff'
_MAC_ZERO_BYTES:      bytes = b'\x00\x00\x00\x00\x00\x00'


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EXCEPTION HIERARCHY
# Typed exceptions carry structured context for agent reasoning.
# ════════════════════════════════════════════════════════════════════════════════

class EtherSentinelError(Exception):
    """Base class for all EtherSentinel errors."""


class PrivilegeError(EtherSentinelError):
    """
    Raised when the process lacks CAP_NET_RAW or root UID.
    Resolution: sudo python3 aios_network.py
               or: sudo setcap cap_net_raw+ep $(which python3)
    """


class InterfaceError(EtherSentinelError):
    """Raised on physical interface open/ioctl/send failure."""


class PolicyViolation(EtherSentinelError):
    """
    Raised when a packet violates the EtherSentinel routing policy.
    EtherSentinel enforces an allowlist; anything not on it is a violation.
    """
    def __init__(
        self,
        direction: str,
        reason:    str,
        frame:     Optional[bytes] = None,
    ) -> None:
        super().__init__(f"PolicyViolation [{direction}]: {reason}")
        self.direction = direction
        self.reason    = reason
        self.frame     = frame


class ARPTimeoutError(EtherSentinelError):
    """Raised when no ARP reply arrives within the configured timeout."""
    def __init__(self, ip: str, timeout: float) -> None:
        super().__init__(
            f"ARP timeout: {ip!r} did not reply within {timeout:.1f}s"
        )
        self.ip      = ip
        self.timeout = timeout


class DNSError(EtherSentinelError):
    """Raised on DNS resolution failure (server-side RCODE or malformed data)."""
    def __init__(self, message: str, rcode: int = 0) -> None:
        super().__init__(message)
        self.rcode = rcode


class DNSTimeoutError(DNSError):
    """Raised when no DNS response arrives within the configured timeout."""
    def __init__(self, name: str, timeout: float) -> None:
        super().__init__(
            f"DNS timeout: {name!r} did not respond within {timeout:.1f}s"
        )
        self.name    = name
        self.timeout = timeout


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TYPED ADDRESS WRAPPERS
# MACAddress and IPv4Address: immutable value objects, O(1) operations.
# ════════════════════════════════════════════════════════════════════════════════

class MACAddress:
    """
    Typed 6-byte MAC address (IEEE 802 EUI-48).

    Accepts bytes, colon-separated hex string, or another MACAddress.
    Equality, hashing, and bytes() conversion are fully supported.
    Two class-level singletons — BROADCAST and ZERO — avoid repeated allocation.
    """
    __slots__ = ('_b',)

    def __init__(self, value: Union[bytes, bytearray, str, 'MACAddress']) -> None:
        if isinstance(value, MACAddress):
            self._b: bytes = value._b
        elif isinstance(value, (bytes, bytearray)):
            if len(value) != 6:
                raise ValueError(
                    f"MAC must be exactly 6 bytes, got {len(value)}"
                )
            self._b = bytes(value)
        elif isinstance(value, str):
            sep = ':' if ':' in value else '-'
            parts = value.split(sep)
            if len(parts) != 6:
                raise ValueError(f"Cannot parse MAC address: {value!r}")
            try:
                self._b = bytes(int(p, 16) for p in parts)
            except ValueError:
                raise ValueError(f"Non-hex octet in MAC address: {value!r}")
        else:
            raise TypeError(
                f"MACAddress requires bytes/str/MACAddress, got {type(value).__name__}"
            )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_broadcast(self) -> bool:
        return self._b == _MAC_BROADCAST_BYTES

    @property
    def is_multicast(self) -> bool:
        """IEEE 802.3: bit 0 of first octet = I/G (individual/group) bit."""
        return bool(self._b[0] & 0x01)

    @property
    def is_locally_administered(self) -> bool:
        """IEEE 802.3: bit 1 of first octet = U/L (universal/local) bit."""
        return bool(self._b[0] & 0x02)

    # ── Dunder ───────────────────────────────────────────────────────────────

    def __bytes__(self) -> bytes:
        return self._b

    def __str__(self) -> str:
        return ':'.join(f'{b:02x}' for b in self._b)

    def __repr__(self) -> str:
        return f'MACAddress("{self}")'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MACAddress):
            return self._b == other._b
        if isinstance(other, (bytes, bytearray)):
            return self._b == bytes(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._b)


# Class-level singletons (created after class body is complete)
MACAddress.BROADCAST: MACAddress = MACAddress(_MAC_BROADCAST_BYTES)  # type: ignore[attr-defined]
MACAddress.ZERO:      MACAddress = MACAddress(_MAC_ZERO_BYTES)        # type: ignore[attr-defined]


class IPv4Address:
    """
    Typed IPv4 address (RFC 791).

    Accepts dotted-decimal string, 4-byte bytes, 32-bit integer, or
    another IPv4Address instance.  Stored internally as a 32-bit unsigned
    integer in host byte order for O(1) arithmetic.
    """
    __slots__ = ('_n',)

    def __init__(
        self,
        value: Union[str, bytes, bytearray, int, 'IPv4Address'],
    ) -> None:
        if isinstance(value, IPv4Address):
            self._n: int = value._n
        elif isinstance(value, str):
            parts = value.split('.')
            if len(parts) != 4:
                raise ValueError(f"Invalid IPv4 address: {value!r}")
            try:
                octets = [int(p) for p in parts]
            except ValueError:
                raise ValueError(f"Non-numeric octet in IPv4: {value!r}")
            if any(not (0 <= o <= 255) for o in octets):
                raise ValueError(f"Octet out of range in IPv4: {value!r}")
            self._n = sum(o << (24 - 8 * i) for i, o in enumerate(octets))
        elif isinstance(value, (bytes, bytearray)):
            if len(value) != 4:
                raise ValueError(
                    f"IPv4 bytes must be exactly 4 bytes, got {len(value)}"
                )
            self._n = struct.unpack('!I', bytes(value))[0]
        elif isinstance(value, int):
            if not (0 <= value <= 0xFFFF_FFFF):
                raise ValueError(f"IPv4 integer out of range: {value}")
            self._n = value
        else:
            raise TypeError(
                f"IPv4Address requires str/bytes/int, got {type(value).__name__}"
            )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def packed(self) -> bytes:
        """Network-byte-order 4-byte representation."""
        return struct.pack('!I', self._n)

    @property
    def as_int(self) -> int:
        return self._n

    def is_loopback(self) -> bool:
        return (self._n >> 24) == 127

    def is_private(self) -> bool:
        """RFC 1918 private address ranges."""
        b0 = (self._n >> 24) & 0xFF
        b1 = (self._n >> 16) & 0xFF
        return (
            b0 == 10
            or (b0 == 172 and 16 <= b1 <= 31)
            or (b0 == 192 and b1 == 168)
        )

    def is_link_local(self) -> bool:
        """RFC 3927 — 169.254.0.0/16."""
        return (self._n >> 16) == 0xA9FE

    def is_broadcast(self) -> bool:
        return self._n == 0xFFFF_FFFF

    # ── Dunder ───────────────────────────────────────────────────────────────

    def __bytes__(self) -> bytes:
        return self.packed

    def __str__(self) -> str:
        return '.'.join(str((self._n >> (24 - 8 * i)) & 0xFF) for i in range(4))

    def __repr__(self) -> str:
        return f'IPv4Address("{self}")'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IPv4Address):
            return self._n == other._n
        if isinstance(other, str):
            try:
                return self._n == IPv4Address(other)._n
            except (ValueError, TypeError):
                return False
        if isinstance(other, int):
            return self._n == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._n)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NETWORK CONFIGURATION
# NetConfig is a shared-mutable object: all subsystems hold a reference to
# the same instance and see mutations (router_mac discovery, host_mac detection)
# without needing notification.
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class NetConfig:
    """
    Complete, explicit configuration for EtherSentinel.

    Every field must be provided by the caller.  No values are ever derived
    from /etc/resolv.conf, the kernel routing table, or network-manager daemons.
    This is a deliberate design decision: EtherSentinel must be fully
    deterministic at initialisation time.

    Mutable fields (router_mac, host_mac) are populated during start() when
    not provided, via ARP discovery and ioctl respectively.  All subsystems
    (PhysicalInterface, PacketFilter, DNSStub) hold a reference to this same
    object and observe mutations live.

    Invariant enforced at construction:
        custom_dns_ip ≠ router_ip  (non-router DNS is architectural)
    """
    interface:     str                          # Linux NIC name, e.g. "eth0"
    host_ip:       Union[str, IPv4Address]      # static IP for this host
    router_ip:     Union[str, IPv4Address]      # gateway / router IP
    custom_dns_ip: Union[str, IPv4Address]      # non-router DNS resolver IP
    subnet_mask:   Union[str, IPv4Address]      # e.g. "255.255.255.0"
    router_mac:    Optional[Union[str, MACAddress]] = None  # None → discover
    host_mac:      Optional[Union[str, MACAddress]] = None  # None → ioctl

    def __post_init__(self) -> None:
        # ── Coerce all str inputs to typed objects ────────────────────────────
        if isinstance(self.host_ip, str):
            self.host_ip = IPv4Address(self.host_ip)
        if isinstance(self.router_ip, str):
            self.router_ip = IPv4Address(self.router_ip)
        if isinstance(self.custom_dns_ip, str):
            self.custom_dns_ip = IPv4Address(self.custom_dns_ip)
        if isinstance(self.subnet_mask, str):
            self.subnet_mask = IPv4Address(self.subnet_mask)
        if isinstance(self.router_mac, str):
            self.router_mac = MACAddress(self.router_mac)
        if isinstance(self.host_mac, str):
            self.host_mac = MACAddress(self.host_mac)

        # ── Structural invariant: DNS ≠ router ───────────────────────────────
        if self.custom_dns_ip == self.router_ip:
            raise ValueError(
                f"custom_dns_ip ({self.custom_dns_ip}) must differ from "
                f"router_ip ({self.router_ip}).  EtherSentinel enforces "
                f"non-router DNS as an architectural invariant — using the "
                f"router's built-in resolver is explicitly disallowed."
            )

    def is_same_subnet(self, ip: IPv4Address) -> bool:
        """
        True if ip falls within the same subnet as host_ip.
        Uses bitwise AND with subnet_mask, per RFC 950.
        """
        mask = self.subnet_mask.as_int
        return (self.host_ip.as_int & mask) == (ip.as_int & mask)

    def __repr__(self) -> str:
        return (
            f"NetConfig("
            f"iface={self.interface!r}, "
            f"host={self.host_ip}, "
            f"router={self.router_ip}, "
            f"dns={self.custom_dns_ip})"
        )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CODEC LAYER
# Pure-static codecs for each protocol layer.  No state.  Every method is a
# bijection: encode(decode(x)) == x for any valid wire-format x.
# ════════════════════════════════════════════════════════════════════════════════

# ── 4.1 Ethernet (IEEE 802.3) ────────────────────────────────────────────────

class EtherCodec:
    """
    Encode and decode IEEE 802.3 Ethernet frames.

    Frame layout (no VLAN tagging):
        Destination MAC  : 6 bytes
        Source MAC       : 6 bytes
        EtherType        : 2 bytes (big-endian)
        Payload          : 46–1500 bytes (padded to ETH_MIN_PAYLOAD if shorter)
        FCS              : 4 bytes — handled by hardware/kernel; NOT included here

    Minimum frame size (excluding FCS): 60 bytes → min payload padded to 46.
    """

    @staticmethod
    def encode(
        dst:       MACAddress,
        src:       MACAddress,
        ethertype: int,
        payload:   bytes,
    ) -> bytes:
        """
        Build an Ethernet frame.
        Pads payload to ETH_MIN_PAYLOAD if necessary.
        Raises ValueError if payload exceeds ETH_MAX_PAYLOAD.
        """
        if len(payload) > ETH_MAX_PAYLOAD:
            raise ValueError(
                f"Payload {len(payload)} B exceeds Ethernet MTU {ETH_MAX_PAYLOAD} B"
            )
        if len(payload) < ETH_MIN_PAYLOAD:
            payload = payload + bytes(ETH_MIN_PAYLOAD - len(payload))
        return bytes(dst) + bytes(src) + struct.pack('!H', ethertype) + payload

    @staticmethod
    def decode(frame: bytes) -> Tuple[MACAddress, MACAddress, int, bytes]:
        """
        Decode an Ethernet frame.
        Returns (dst_mac, src_mac, ethertype, payload).
        Raises ValueError on truncated frame.
        """
        if len(frame) < ETH_HEADER_LEN:
            raise ValueError(
                f"Frame too short: {len(frame)} < {ETH_HEADER_LEN} bytes"
            )
        dst       = MACAddress(frame[0:6])
        src       = MACAddress(frame[6:12])
        ethertype = struct.unpack('!H', frame[12:14])[0]
        payload   = frame[14:]
        return dst, src, ethertype, payload


# ── 4.2 ARP (RFC 826) ────────────────────────────────────────────────────────

@dataclass
class ARPPacket:
    """Decoded ARP packet for Ethernet/IPv4 (HTYPE=1, PTYPE=0x0800)."""
    operation:  int           # ARP_OP_REQUEST or ARP_OP_REPLY
    sender_mac: MACAddress
    sender_ip:  IPv4Address
    target_mac: MACAddress
    target_ip:  IPv4Address


class ARPCodec:
    """
    Encode and decode ARP packets per RFC 826.

    Wire layout (Ethernet/IPv4 only):
        HTYPE  : 2 bytes — 1 (Ethernet)
        PTYPE  : 2 bytes — 0x0800 (IPv4)
        HLEN   : 1 byte  — 6
        PLEN   : 1 byte  — 4
        OPER   : 2 bytes — 1=request, 2=reply
        SHA    : 6 bytes — sender hardware address
        SPA    : 4 bytes — sender protocol address
        THA    : 6 bytes — target hardware address
        TPA    : 4 bytes — target protocol address
    Total: 28 bytes
    """

    @staticmethod
    def encode(pkt: ARPPacket) -> bytes:
        """Serialise an ARPPacket to 28-byte wire format."""
        return struct.pack(
            '!HHBBH',
            ARP_HW_ETHERNET,  # HTYPE
            ARP_PROTO_IPv4,   # PTYPE
            6,                # HLEN
            4,                # PLEN
            pkt.operation,    # OPER
        ) + bytes(pkt.sender_mac) + bytes(pkt.sender_ip) \
          + bytes(pkt.target_mac) + bytes(pkt.target_ip)

    @staticmethod
    def decode(data: bytes) -> ARPPacket:
        """
        Deserialise 28-byte ARP wire format into an ARPPacket.
        Raises ValueError for unsupported hardware/protocol types or truncation.
        """
        if len(data) < ARP_PACKET_LEN:
            raise ValueError(
                f"ARP payload too short: {len(data)} < {ARP_PACKET_LEN}"
            )
        htype, ptype, hlen, plen, op = struct.unpack('!HHBBH', data[:8])
        if htype != ARP_HW_ETHERNET:
            raise ValueError(f"Unsupported ARP HTYPE: {htype:#06x} (expected Ethernet=1)")
        if ptype != ARP_PROTO_IPv4:
            raise ValueError(f"Unsupported ARP PTYPE: {ptype:#06x} (expected IPv4=0x0800)")
        if hlen != 6:
            raise ValueError(f"Unexpected ARP HLEN: {hlen} (expected 6)")
        if plen != 4:
            raise ValueError(f"Unexpected ARP PLEN: {plen} (expected 4)")
        return ARPPacket(
            operation=op,
            sender_mac=MACAddress(data[8:14]),
            sender_ip=IPv4Address(data[14:18]),
            target_mac=MACAddress(data[18:24]),
            target_ip=IPv4Address(data[24:28]),
        )


# ── 4.3 IPv4 (RFC 791) ───────────────────────────────────────────────────────

@dataclass
class IPv4Packet:
    """Decoded IPv4 packet.  Options are not supported (IHL must be 5)."""
    src_ip:         IPv4Address
    dst_ip:         IPv4Address
    protocol:       int           # IPPROTO_UDP, IPPROTO_TCP, …
    payload:        bytes
    ttl:            int = 64
    identification: int = 0
    flags_offset:   int = IP_FLAG_DF   # DF set, no fragmentation


class IPv4Codec:
    """
    Encode and decode IPv4 packets per RFC 791.

    Header layout (no options, IHL=5):
        Version+IHL : 1 byte  — 0x45 (v4, 20-byte header)
        DSCP+ECN    : 1 byte  — 0x00
        Total Length: 2 bytes — header + payload
        ID          : 2 bytes
        Flags+Frag  : 2 bytes — DF bit set, fragment offset = 0
        TTL         : 1 byte
        Protocol    : 1 byte
        Checksum    : 2 bytes — RFC 1071 one's complement sum
        Src IP      : 4 bytes
        Dst IP      : 4 bytes
    Total header: 20 bytes

    Checksum algorithm (RFC 1071):
        Sum all 16-bit words of the header (checksum field zeroed).
        Fold 32-bit carry into 16 bits.
        One's complement of final 16-bit sum.
        Receiver verifies: sum of all 16-bit words (including checksum) == 0xFFFF.
    """

    @staticmethod
    def checksum(data: bytes) -> int:
        """
        RFC 1071 internet checksum.
        data must be the header bytes with the checksum field set to 0x0000.
        Returns the 16-bit one's complement checksum.
        """
        if len(data) % 2:
            data = data + b'\x00'          # pad to even length
        total: int = sum(struct.unpack(f'!{len(data)//2}H', data))
        while total >> 16:                 # fold any 32-bit carry
            total = (total & 0xFFFF) + (total >> 16)
        return (~total) & 0xFFFF           # one's complement

    @staticmethod
    def encode(pkt: IPv4Packet) -> bytes:
        """
        Serialise an IPv4Packet.  Computes and embeds the header checksum.
        Total length = 20 (header) + len(payload).
        """
        total_len = IP_HEADER_MIN_LEN + len(pkt.payload)
        # Build header with checksum = 0 for computation
        header = struct.pack(
            '!BBHHHBBH4s4s',
            IP_VERSION_IHL,           # version=4, IHL=5
            0,                        # DSCP=0, ECN=0
            total_len,
            pkt.identification,
            pkt.flags_offset,
            pkt.ttl,
            pkt.protocol,
            0,                        # checksum placeholder
            bytes(pkt.src_ip),
            bytes(pkt.dst_ip),
        )
        cksum = IPv4Codec.checksum(header)
        # Splice computed checksum into bytes 10–11
        return header[:10] + struct.pack('!H', cksum) + header[12:] + pkt.payload

    @staticmethod
    def decode(data: bytes) -> IPv4Packet:
        """
        Deserialise an IPv4 datagram.
        Verifies header checksum.  Raises ValueError on any structural error.
        Does not support fragmented datagrams (fragment offset must be 0).
        """
        if len(data) < IP_HEADER_MIN_LEN:
            raise ValueError(
                f"IPv4 datagram too short: {len(data)} < {IP_HEADER_MIN_LEN}"
            )
        ver_ihl = data[0]
        version = (ver_ihl >> 4) & 0x0F
        ihl     = (ver_ihl & 0x0F) * 4      # in bytes

        if version != 4:
            raise ValueError(f"Not IPv4: version={version}")
        if ihl < IP_HEADER_MIN_LEN:
            raise ValueError(f"IPv4 IHL too small: {ihl} bytes")
        if ihl > len(data):
            raise ValueError(
                f"IPv4 IHL {ihl} exceeds datagram length {len(data)}"
            )

        total_len = struct.unpack('!H', data[2:4])[0]
        if total_len > len(data):
            raise ValueError(
                f"IPv4 total_len {total_len} exceeds buffer {len(data)}"
            )

        # ── Header checksum verification ──────────────────────────────────────
        header_bytes = data[:ihl]
        received_cksum = struct.unpack('!H', header_bytes[10:12])[0]
        check_header   = header_bytes[:10] + b'\x00\x00' + header_bytes[12:]
        computed_cksum  = IPv4Codec.checksum(check_header)
        if computed_cksum != received_cksum:
            raise ValueError(
                f"IPv4 checksum mismatch: "
                f"received={received_cksum:#06x}, computed={computed_cksum:#06x}"
            )

        # ── Fragment check ────────────────────────────────────────────────────
        flags_offset = struct.unpack('!H', data[6:8])[0]
        if flags_offset & 0x1FFF:   # fragment offset ≠ 0
            raise ValueError("Fragmented IPv4 datagrams are not supported")

        ident, _, ttl, proto = struct.unpack('!HHBB', data[4:10])
        src_ip = IPv4Address(data[12:16])
        dst_ip = IPv4Address(data[16:20])

        return IPv4Packet(
            src_ip=src_ip,
            dst_ip=dst_ip,
            protocol=proto,
            payload=data[ihl:total_len],
            ttl=ttl,
            identification=ident,
            flags_offset=flags_offset,
        )


# ── 4.4 UDP (RFC 768) ────────────────────────────────────────────────────────

@dataclass
class UDPDatagram:
    """Decoded UDP datagram."""
    src_port: int
    dst_port: int
    payload:  bytes


class UDPCodec:
    """
    Encode and decode UDP datagrams per RFC 768.

    Header layout:
        Source Port      : 2 bytes
        Destination Port : 2 bytes
        Length           : 2 bytes — header (8) + payload
        Checksum         : 2 bytes — set to 0 (optional in IPv4, RFC 768 §4)

    UDP checksum is disabled (0) in this implementation.  IPv4 does not
    require UDP checksum; the IPv4 header checksum covers the IP layer.
    End-to-end integrity is ensured at the application (DNS) layer by the
    query ID correlation mechanism.
    """

    @staticmethod
    def encode(dgram: UDPDatagram) -> bytes:
        """Serialise a UDPDatagram.  Checksum field is set to 0."""
        length = UDP_HEADER_LEN + len(dgram.payload)
        if length > 65535:
            raise ValueError(f"UDP datagram too large: {length} bytes")
        return struct.pack(
            '!HHHH',
            dgram.src_port,
            dgram.dst_port,
            length,
            0,          # checksum = 0 (disabled)
        ) + dgram.payload

    @staticmethod
    def decode(data: bytes) -> UDPDatagram:
        """
        Deserialise a UDP datagram.
        Raises ValueError on truncation or invalid length field.
        """
        if len(data) < UDP_HEADER_LEN:
            raise ValueError(
                f"UDP datagram too short: {len(data)} < {UDP_HEADER_LEN}"
            )
        src_port, dst_port, length, _ = struct.unpack('!HHHH', data[:8])
        if length < UDP_HEADER_LEN:
            raise ValueError(f"UDP length field {length} < header {UDP_HEADER_LEN}")
        if length > len(data):
            raise ValueError(
                f"UDP length field {length} exceeds available data {len(data)}"
            )
        return UDPDatagram(
            src_port=src_port,
            dst_port=dst_port,
            payload=data[8:length],
        )


# ── 4.5 DNS (RFC 1035) ───────────────────────────────────────────────────────

@dataclass
class DNSRecord:
    """Single resource record from a DNS response."""
    name:  str
    rtype: int   # DNS_TYPE_A, DNS_TYPE_AAAA, DNS_TYPE_CNAME, …
    ttl:   int
    value: str   # human-readable: dotted-decimal IP, IPv6, FQDN, or raw hex


@dataclass
class DNSResponse:
    """Decoded DNS response message."""
    query_id: int
    flags:    int
    answers:  List[DNSRecord]

    @property
    def rcode(self) -> int:
        return self.flags & 0x000F

    @property
    def is_authoritative(self) -> bool:
        return bool(self.flags & DNS_FLAG_AA)

    @property
    def is_truncated(self) -> bool:
        return bool(self.flags & DNS_FLAG_TC)

    @property
    def recursion_available(self) -> bool:
        return bool(self.flags & DNS_FLAG_RA)

    def a_records(self) -> List[str]:
        return [r.value for r in self.answers if r.rtype == DNS_TYPE_A]

    def aaaa_records(self) -> List[str]:
        return [r.value for r in self.answers if r.rtype == DNS_TYPE_AAAA]

    def cname_records(self) -> List[str]:
        return [r.value for r in self.answers if r.rtype == DNS_TYPE_CNAME]


class DNSCodec:
    """
    Encode and decode DNS messages per RFC 1035.

    Message layout:
        Header section  : 12 bytes (ID, FLAGS, QDCOUNT, ANCOUNT, NSCOUNT, ARCOUNT)
        Question section: variable (QNAME, QTYPE, QCLASS)
        Answer section  : variable (NAME, TYPE, CLASS, TTL, RDLENGTH, RDATA)
        Authority section: skipped
        Additional section: skipped

    Name encoding (§3.1):
        Labels separated by length-prefixed octets, terminated by 0x00.
        Compression (§4.1.4): 2-byte pointer with top 2 bits = 11 (0xC0).

    Supported RDATA types:
        A     (type  1) : 4-byte IPv4 address → dotted decimal string
        NS    (type  2) : domain name (stored as-is)
        CNAME (type  5) : domain name (followed for resolution)
        MX    (type 15) : preference + domain (stored as "pref:name")
        AAAA  (type 28) : 16-byte IPv6 address → colon-hex string
        Other           : raw hexadecimal string
    """

    # RFC 1035 RCODE names for error reporting
    _RCODE_NAMES: Dict[int, str] = {
        0: 'NOERROR',
        1: 'FORMERR',
        2: 'SERVFAIL',
        3: 'NXDOMAIN',
        4: 'NOTIMP',
        5: 'REFUSED',
    }

    @staticmethod
    def encode_name(name: str) -> bytes:
        """
        Encode a domain name to DNS wire-format label sequence (RFC 1035 §3.1).
        Each label: 1-byte length + ASCII octets.  Terminated by 0x00 root label.
        Maximum label length: 63 octets.
        """
        result = b''
        for label in name.rstrip('.').split('.'):
            if not label:
                continue
            encoded_label = label.encode('ascii')
            if len(encoded_label) > 63:
                raise ValueError(
                    f"DNS label exceeds 63 bytes: {label!r} ({len(encoded_label)} bytes)"
                )
            result += struct.pack('B', len(encoded_label)) + encoded_label
        return result + b'\x00'   # root label

    @staticmethod
    def encode_query(query_id: int, name: str, qtype: int = DNS_TYPE_A) -> bytes:
        """
        Build a standard DNS query packet (RFC 1035 §4.1).

        Header flags: QR=0 (query), OPCODE=0 (standard), RD=1 (recursion desired).
        One question, zero answers/authority/additional sections.
        """
        header = struct.pack(
            '!HHHHHH',
            query_id,
            DNS_FLAG_RD,   # QR=0, OPCODE=0, AA=0, TC=0, RD=1
            1,             # QDCOUNT: one question
            0,             # ANCOUNT
            0,             # NSCOUNT
            0,             # ARCOUNT
        )
        question = (
            DNSCodec.encode_name(name)
            + struct.pack('!HH', qtype, DNS_CLASS_IN)
        )
        return header + question

    @staticmethod
    def decode_name(data: bytes, offset: int) -> Tuple[str, int]:
        """
        Decode a DNS name from wire format, following compression pointers.
        (RFC 1035 §4.1.4)

        Returns (name_str, next_offset_after_name).
        next_offset_after_name is the byte immediately following the name
        in the non-compressed path (the first pointer, if compression was used).

        Raises ValueError on truncation or compression loop (> 20 hops).
        """
        labels:      List[str] = []
        jumped:      bool      = False
        orig_offset: int       = offset
        jumps:       int       = 0
        max_jumps:   int       = 20

        while offset < len(data):
            length = data[offset]

            if length == 0:                        # root label → end of name
                if not jumped:
                    orig_offset = offset + 1
                break

            if (length & 0xC0) == 0xC0:            # compression pointer
                if offset + 1 >= len(data):
                    raise ValueError(
                        f"DNS: truncated compression pointer at offset {offset}"
                    )
                pointer = ((length & 0x3F) << 8) | data[offset + 1]
                if not jumped:
                    orig_offset = offset + 2       # save post-pointer return point
                offset  = pointer
                jumped  = True
                jumps  += 1
                if jumps > max_jumps:
                    raise ValueError(
                        "DNS: compression loop detected (> 20 pointer hops)"
                    )
                continue

            if (length & 0xC0) != 0:               # reserved top-bit patterns
                raise ValueError(
                    f"DNS: unsupported label type {length & 0xC0:#04x} at offset {offset}"
                )

            # Ordinary label
            offset += 1
            end     = offset + length
            if end > len(data):
                raise ValueError(
                    f"DNS: label at offset {offset - 1} extends past data end "
                    f"(need {end}, have {len(data)})"
                )
            labels.append(data[offset:end].decode('ascii', errors='replace'))
            offset = end

        if not jumped:
            orig_offset = offset + 1

        return '.'.join(labels), orig_offset

    @staticmethod
    def decode_response(data: bytes) -> DNSResponse:
        """
        Decode a complete DNS response message (RFC 1035 §4).

        Parses header, skips question section, parses all answer RRs.
        Raises DNSError for server-returned RCODE errors.
        Raises ValueError for structural problems (truncation, bad lengths).
        """
        if len(data) < 12:
            raise ValueError(
                f"DNS response too short: {len(data)} bytes (need ≥ 12 for header)"
            )

        qid, flags, qdcount, ancount, nscount, arcount = \
            struct.unpack('!HHHHHH', data[:12])

        rcode = flags & 0x000F
        if rcode != 0:
            rcode_name = DNSCodec._RCODE_NAMES.get(rcode, f'RCODE={rcode}')
            raise DNSError(
                f"DNS server returned {rcode_name}",
                rcode,
            )

        offset = 12

        # ── Skip question section ─────────────────────────────────────────────
        for _ in range(qdcount):
            _, offset = DNSCodec.decode_name(data, offset)
            if offset + 4 > len(data):
                raise ValueError("DNS: truncated question section (need QTYPE+QCLASS)")
            offset += 4       # QTYPE(2) + QCLASS(2)

        # ── Parse answer section ──────────────────────────────────────────────
        answers: List[DNSRecord] = []
        for _ in range(ancount):
            if offset >= len(data):
                break

            name, offset = DNSCodec.decode_name(data, offset)

            if offset + 10 > len(data):
                raise ValueError(
                    f"DNS: truncated answer RR at offset {offset} "
                    f"(need 10 bytes for TYPE+CLASS+TTL+RDLENGTH)"
                )

            rtype, rclass, ttl, rdlength = struct.unpack(
                '!HHIH', data[offset:offset + 10]
            )
            offset += 10

            if offset + rdlength > len(data):
                raise ValueError(
                    f"DNS: RDATA at offset {offset} exceeds data "
                    f"(RDLENGTH={rdlength}, available={len(data)-offset})"
                )

            rdata       = data[offset:offset + rdlength]
            rdata_start = offset
            offset     += rdlength

            # ── Decode RDATA by type ──────────────────────────────────────────
            if rtype == DNS_TYPE_A and rdlength == 4:
                value = '.'.join(str(b) for b in rdata)

            elif rtype == DNS_TYPE_AAAA and rdlength == 16:
                # RFC 5952 canonical IPv6 representation (simplified: no :: compression)
                groups = [
                    f'{(rdata[i] << 8 | rdata[i + 1]):04x}'
                    for i in range(0, 16, 2)
                ]
                value = ':'.join(groups)

            elif rtype == DNS_TYPE_CNAME:
                cname, _ = DNSCodec.decode_name(data, rdata_start)
                value = cname

            elif rtype == DNS_TYPE_NS:
                ns, _ = DNSCodec.decode_name(data, rdata_start)
                value = ns

            elif rtype == DNS_TYPE_MX and rdlength >= 3:
                preference = struct.unpack('!H', rdata[:2])[0]
                exchange, _ = DNSCodec.decode_name(data, rdata_start + 2)
                value = f'{preference}:{exchange}'

            else:
                value = rdata.hex()   # unknown type: raw hex string

            answers.append(DNSRecord(
                name=name,
                rtype=rtype,
                ttl=ttl,
                value=value,
            ))

        return DNSResponse(query_id=qid, flags=flags, answers=answers)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PACKET FILTER
# Strict outbound + inbound allowlist.  Counts every frame class separately.
# The policy is derived exclusively from NetConfig; no dynamic rules.
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class _FilterStats:
    outbound_allowed: int = 0
    outbound_blocked: int = 0
    inbound_allowed:  int = 0
    inbound_blocked:  int = 0


class PacketFilter:
    """
    Allowlist-based packet filter for EtherSentinel.

    Outbound rules (checked before any send):
        ARP  → allowed only if dst_mac is BROADCAST or router_mac
        UDP  → allowed only if dst_ip == custom_dns_ip AND dst_port == 53

    Inbound rules (checked in the receive dispatch loop):
        ARP  → allowed if src_mac == router_mac
               (or router_mac is None, during initial router MAC discovery)
        UDP  → allowed only if src_ip == custom_dns_ip AND src_port == 53

    All other frames are silently counted and discarded.
    Violation counters are exposed via the stats property.
    """

    def __init__(self, config: NetConfig) -> None:
        self._cfg   = config           # shared mutable reference
        self._stats = _FilterStats()
        self._lock  = threading.Lock()

    def check_outbound_arp(self, dst_mac: MACAddress) -> bool:
        """
        True if the ARP frame may be sent.
        Broadcast (router discovery) and unicast to router_mac are permitted.
        """
        allowed = dst_mac.is_broadcast or dst_mac == self._cfg.router_mac
        with self._lock:
            if allowed:
                self._stats.outbound_allowed += 1
            else:
                self._stats.outbound_blocked += 1
        return allowed

    def check_outbound_udp(self, dst_ip: IPv4Address, dst_port: int) -> bool:
        """
        True if the UDP datagram may be sent.
        Only DNS queries to custom_dns_ip:53 are permitted.
        """
        allowed = (dst_ip == self._cfg.custom_dns_ip and dst_port == DNS_PORT)
        with self._lock:
            if allowed:
                self._stats.outbound_allowed += 1
            else:
                self._stats.outbound_blocked += 1
        return allowed

    def check_inbound_arp(self, src_mac: MACAddress) -> bool:
        """
        True if the inbound ARP frame should be processed.
        Before router_mac is known (None), any ARP reply is accepted
        to enable initial router MAC discovery.
        After discovery, only frames from router_mac are accepted.
        """
        if self._cfg.router_mac is None:
            # Pre-discovery: accept any ARP to learn the router MAC
            allowed = True
        else:
            allowed = (src_mac == self._cfg.router_mac)
        with self._lock:
            if allowed:
                self._stats.inbound_allowed += 1
            else:
                self._stats.inbound_blocked += 1
        return allowed

    def check_inbound_ip(
        self,
        src_ip:   IPv4Address,
        protocol: int,
        src_port: Optional[int] = None,
    ) -> bool:
        """
        True if the inbound IP packet should be processed.
        Only UDP from custom_dns_ip:53 is accepted.
        """
        allowed = (
            protocol == IPPROTO_UDP
            and src_ip   == self._cfg.custom_dns_ip
            and src_port == DNS_PORT
        )
        with self._lock:
            if allowed:
                self._stats.inbound_allowed += 1
            else:
                self._stats.inbound_blocked += 1
        return allowed

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                'outbound_allowed': self._stats.outbound_allowed,
                'outbound_blocked': self._stats.outbound_blocked,
                'inbound_allowed':  self._stats.inbound_allowed,
                'inbound_blocked':  self._stats.inbound_blocked,
            }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ARP CACHE
# Thread-safe LRU-style ARP table with per-entry TTL and resolution wait.
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class _ARPEntry:
    mac:        MACAddress
    created_at: float

    def is_expired(self, ttl: float) -> bool:
        return (time.monotonic() - self.created_at) > ttl


class ARPCache:
    """
    Thread-safe ARP resolution cache.

    Entries expire after ttl_seconds (default ARP_CACHE_TTL_S = 300 s).
    The wait_for() method blocks the calling thread until the requested IP
    appears in the cache or the timeout expires, allowing the receive loop to
    populate it asynchronously via set().

    All public methods are safe to call from any thread.
    """

    def __init__(self, ttl_seconds: float = ARP_CACHE_TTL_S) -> None:
        self._ttl:     float                     = ttl_seconds
        self._table:   Dict[str, _ARPEntry]      = {}
        self._lock:    threading.RLock           = threading.RLock()
        self._waiters: Dict[str, threading.Event] = {}

    def get(self, ip: str) -> Optional[MACAddress]:
        """
        Return cached MAC for ip, or None if absent / expired.
        Evicts expired entries on access (lazy expiry).
        """
        with self._lock:
            entry = self._table.get(ip)
            if entry is None:
                return None
            if entry.is_expired(self._ttl):
                del self._table[ip]
                return None
            return entry.mac

    def set(self, ip: str, mac: MACAddress) -> None:
        """
        Insert or update a mapping.  Wakes any thread waiting for this IP.
        """
        with self._lock:
            self._table[ip] = _ARPEntry(mac=mac, created_at=time.monotonic())
            ev = self._waiters.get(ip)
        # Signal outside the lock to avoid priority inversion
        if ev is not None:
            ev.set()

    def wait_for(self, ip: str, timeout: float) -> Optional[MACAddress]:
        """
        Block until ip is in the cache or timeout expires.
        If already cached (and not expired), returns immediately.
        Returns the MACAddress on success, None on timeout.
        """
        ev = threading.Event()
        with self._lock:
            existing = self.get(ip)
            if existing is not None:
                return existing
            self._waiters[ip] = ev

        ev.wait(timeout=timeout)

        with self._lock:
            self._waiters.pop(ip, None)
            return self.get(ip)

    def invalidate(self, ip: str) -> None:
        """Force removal of an entry, ignoring TTL."""
        with self._lock:
            self._table.pop(ip, None)

    def purge_expired(self) -> int:
        """Remove all TTL-expired entries.  Returns count removed."""
        with self._lock:
            before = len(self._table)
            self._table = {
                ip: entry
                for ip, entry in self._table.items()
                if not entry.is_expired(self._ttl)
            }
            return before - len(self._table)

    def as_dict(self) -> Dict[str, str]:
        """Snapshot of live (non-expired) ARP table as {ip_str: mac_str}."""
        with self._lock:
            return {
                ip: str(entry.mac)
                for ip, entry in self._table.items()
                if not entry.is_expired(self._ttl)
            }

    def __len__(self) -> int:
        with self._lock:
            return sum(
                1 for e in self._table.values()
                if not e.is_expired(self._ttl)
            )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PHYSICAL INTERFACE
# Raw AF_PACKET socket on Linux.  The kernel IP stack is entirely bypassed:
# we receive and send complete Ethernet frames, including the header.
# ════════════════════════════════════════════════════════════════════════════════

FrameHandler = Callable[[MACAddress, MACAddress, bytes], None]
"""
Type alias for frame dispatch callbacks.
Signature: handler(src_mac: MACAddress, dst_mac: MACAddress, payload: bytes) → None
"""


class PhysicalInterface:
    """
    AF_PACKET raw socket layer for Linux.

    Opens the named network interface at Ethernet (Layer 2) level,
    bypassing the kernel IP stack completely.  Ethernet frames are received
    and transmitted in their entirety (excluding FCS, which the hardware/kernel
    appends transparently).

    The receive loop runs in a single daemon thread named "EtherSentinel-RX".
    Frame dispatch is type-based: handlers registered via register_handler()
    are called synchronously in the RX thread for each matching EtherType.
    Long handler operations should offload to a work queue; the RX thread
    must remain fast to prevent socket buffer overflow.

    Thread safety:
        send_frame()         — protected by _tx_lock
        register_handler()   — protected by _lock
        _receive_loop()      — sole reader of _sock

    Platform requirement: Linux ≥ 3.0, CAP_NET_RAW or UID 0.
    """

    def __init__(self, config: NetConfig) -> None:
        if platform.system() != 'Linux':
            raise InterfaceError(
                f"EtherSentinel requires Linux (AF_PACKET).  "
                f"Current platform: {platform.system()!r}.  "
                f"BSD/macOS BPF is not supported in this implementation."
            )
        self._cfg:      NetConfig                        = config
        self._sock:     Optional[socket.socket]          = None
        self._running:  bool                             = False
        self._rx_thread: Optional[threading.Thread]      = None
        self._handlers: Dict[int, List[FrameHandler]]    = defaultdict(list)
        self._lock:     threading.Lock                   = threading.Lock()
        self._tx_lock:  threading.Lock                   = threading.Lock()
        self._stats:    Dict[str, int]                   = defaultdict(int)

    # ── Privilege check ───────────────────────────────────────────────────────

    def _check_privileges(self) -> None:
        """
        Raise PrivilegeError if the process cannot open AF_PACKET sockets.
        Checks UID == 0 first; then reads /proc/self/status for CAP_NET_RAW
        (capability bit 13 in the CapPrm bitmask).
        """
        if os.geteuid() == 0:
            return   # root: unconditionally allowed

        try:
            with open('/proc/self/status', 'r') as fh:
                for line in fh:
                    if line.startswith('CapPrm:'):
                        cap_bitmask = int(line.split()[1], 16)
                        CAP_NET_RAW_BIT = 1 << 13
                        if cap_bitmask & CAP_NET_RAW_BIT:
                            return   # CAP_NET_RAW present
        except OSError:
            pass   # /proc not available — fall through to error

        raise PrivilegeError(
            "EtherSentinel requires CAP_NET_RAW or UID 0 to open AF_PACKET sockets.\n"
            "  Option 1: sudo python3 aios_network.py\n"
            "  Option 2: sudo setcap cap_net_raw+ep $(which python3)"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open the AF_PACKET socket and bind it to the configured interface.
        Auto-detects host MAC via SIOCGIFHWADDR if not supplied in config.
        Sets socket timeout to RX_SOCKET_TIMEOUT for clean shutdown polling.
        """
        self._check_privileges()

        try:
            self._sock = socket.socket(
                socket.AF_PACKET,
                socket.SOCK_RAW,
                socket.htons(ETH_P_ALL),
            )
            self._sock.bind((self._cfg.interface, 0))
            self._sock.settimeout(RX_SOCKET_TIMEOUT)
        except OSError as exc:
            raise InterfaceError(
                f"Cannot open AF_PACKET on interface {self._cfg.interface!r}: {exc}"
            ) from exc

        if self._cfg.host_mac is None:
            self._cfg.host_mac = self._read_hw_addr()

    def _read_hw_addr(self) -> MACAddress:
        """
        Read the hardware (MAC) address of the bound interface via ioctl.
        Uses SIOCGIFHWADDR (0x8927) from <linux/sockios.h>.
        ifreq layout: interface name (16 bytes) followed by sockaddr (14 bytes).
        The MAC occupies bytes 18–23 of the returned ifreq.
        """
        ifreq = struct.pack('256s', self._cfg.interface.encode()[:15])
        try:
            result = fcntl.ioctl(self._sock.fileno(), SIOCGIFHWADDR, ifreq)
        except OSError as exc:
            raise InterfaceError(
                f"SIOCGIFHWADDR failed for {self._cfg.interface!r}: {exc}"
            ) from exc
        return MACAddress(result[18:24])

    def close(self) -> None:
        """
        Signal the RX loop to exit and close the socket.
        Blocks up to 2× RX_SOCKET_TIMEOUT for a clean thread join.
        Safe to call multiple times.
        """
        self._running = False
        if self._rx_thread is not None and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=RX_SOCKET_TIMEOUT * 2)
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def is_open(self) -> bool:
        return self._sock is not None

    # ── Frame I/O ─────────────────────────────────────────────────────────────

    def send_frame(self, frame: bytes) -> int:
        """
        Transmit a raw Ethernet frame.
        Returns the number of bytes sent.
        Raises InterfaceError if the socket is not open or send fails.
        Thread-safe (serialised by _tx_lock).
        """
        if self._sock is None:
            raise InterfaceError(
                "Cannot send: interface not open.  Call open() first."
            )
        with self._tx_lock:
            try:
                sent = self._sock.send(frame)
                self._stats['tx_frames'] += 1
                self._stats['tx_bytes']  += sent
                return sent
            except OSError as exc:
                self._stats['tx_errors'] += 1
                raise InterfaceError(f"send_frame failed: {exc}") from exc

    def register_handler(self, ethertype: int, handler: FrameHandler) -> None:
        """
        Register a callback to receive frames of the given EtherType.
        Multiple handlers per EtherType are supported; they are called in
        registration order within the same RX thread turn.
        """
        with self._lock:
            self._handlers[ethertype].append(handler)

    def start_receive_loop(self) -> None:
        """Launch the background RX daemon thread."""
        self._running   = True
        self._rx_thread = threading.Thread(
            target=self._receive_loop,
            name='EtherSentinel-RX',
            daemon=True,
        )
        self._rx_thread.start()

    def _receive_loop(self) -> None:
        """
        Main receive loop executed in the EtherSentinel-RX daemon thread.

        Per iteration:
            1. recvfrom() with timeout → (frame_bytes, ancdata)
               ancdata = (iface_name, proto, pkttype, hatype, hw_addr)
            2. Accept only PACKET_HOST and PACKET_BROADCAST (discard our own TX,
               multicast noise, and promiscuous-mode foreign frames).
            3. Decode Ethernet header via EtherCodec.
            4. Dispatch payload to all registered handlers for the EtherType.

        The loop exits when self._running is False or the socket is closed
        (EBADF / EINVAL on recv).
        """
        while self._running:
            try:
                frame, ancdata = self._sock.recvfrom(65536)
                _, _proto, pkttype, _hatype, _hw_addr = ancdata

                if pkttype not in (PACKET_HOST, PACKET_BROADCAST):
                    self._stats['rx_filtered_pkttype'] += 1
                    continue

                self._stats['rx_frames'] += 1
                self._stats['rx_bytes']  += len(frame)

                if len(frame) < ETH_HEADER_LEN:
                    self._stats['rx_short'] += 1
                    continue

                try:
                    dst_mac, src_mac, ethertype, payload = EtherCodec.decode(frame)
                except ValueError:
                    self._stats['rx_malformed_eth'] += 1
                    continue

                with self._lock:
                    handlers = list(self._handlers.get(ethertype, []))

                if not handlers:
                    self._stats['rx_unhandled_ethertype'] += 1
                    continue

                for handler in handlers:
                    try:
                        handler(src_mac, dst_mac, payload)
                    except Exception:
                        self._stats['rx_handler_errors'] += 1

            except socket.timeout:
                continue   # normal: poll self._running

            except OSError as exc:
                if exc.errno in (errno.EBADF, errno.EINVAL):
                    break  # socket was closed by close()
                self._stats['rx_errors'] += 1

            except Exception:
                self._stats['rx_errors'] += 1

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def host_mac(self) -> Optional[MACAddress]:
        return self._cfg.host_mac

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DNS STUB RESOLVER
# Sends DNS queries via raw UDP/IP/Ethernet to custom_dns_ip:53 exclusively.
# No getaddrinfo(). No /etc/resolv.conf. No system resolver.
# Concurrent queries are safe: each query registers its own (query_id → Event).
# ════════════════════════════════════════════════════════════════════════════════

class DNSStub:
    """
    DNS stub resolver for EtherSentinel.

    Query lifecycle:
        1. resolve(name) generates a random 16-bit query_id and src_port.
        2. A threading.Event is registered in _pending[query_id] before send.
        3. A UDP/IP/Ethernet frame is sent to router MAC (Layer 2 gateway),
           with dst_ip=custom_dns_ip, dst_port=53.
        4. The EtherSentinel-RX thread receives the DNS response, decodes
           it, and calls handle_udp_payload() from the dispatch chain.
        5. handle_udp_payload() finds the matching pending entry by query_id
           and signals the Event.
        6. resolve() unblocks, reads the DNSResponse, and returns IP list.

    CNAME following is recursive (up to DNS_MAX_RETRIES depth total).
    Multiple concurrent resolve() calls are fully safe.
    """

    def __init__(
        self,
        config:  NetConfig,
        iface:   PhysicalInterface,
        arp:     ARPCache,
        pfilter: PacketFilter,
    ) -> None:
        self._cfg:    NetConfig         = config
        self._iface:  PhysicalInterface = iface
        self._arp:    ARPCache          = arp
        self._filter: PacketFilter      = pfilter
        self._lock:   threading.Lock    = threading.Lock()
        # Maps DNS query_id → (Event, List[DNSResponse])
        self._pending: Dict[int, Tuple[threading.Event, List[DNSResponse]]] = {}
        self._next_src_port_val: int = random.randint(
            EPHEMERAL_PORT_LO, EPHEMERAL_PORT_HI
        )
        self._ip_id_counter: int = random.randint(0, 0xFFFF)

    def _allocate_src_port(self) -> int:
        """
        Allocate the next ephemeral UDP source port in the range
        [EPHEMERAL_PORT_LO, EPHEMERAL_PORT_HI] (RFC 6335 §6).
        Wraps around at EPHEMERAL_PORT_HI.
        """
        with self._lock:
            port = self._next_src_port_val
            self._next_src_port_val = (
                (self._next_src_port_val - EPHEMERAL_PORT_LO + 1)
                % (EPHEMERAL_PORT_HI - EPHEMERAL_PORT_LO + 1)
                + EPHEMERAL_PORT_LO
            )
            return port

    def _allocate_ip_id(self) -> int:
        """Allocate next IPv4 identification field value (monotonic, 16-bit wrap)."""
        with self._lock:
            ident = self._ip_id_counter
            self._ip_id_counter = (self._ip_id_counter + 1) & 0xFFFF
            return ident

    def handle_udp_payload(
        self,
        src_ip:   IPv4Address,
        src_port: int,
        payload:  bytes,
    ) -> None:
        """
        Called from the EtherSentinel-RX thread when a UDP packet arrives
        from custom_dns_ip:53.  Attempts to decode the payload as a DNS
        response and signals any matching pending waiter.
        """
        if src_ip != self._cfg.custom_dns_ip or src_port != DNS_PORT:
            return   # not from our DNS server

        try:
            response = DNSCodec.decode_response(payload)
        except (DNSError, ValueError, struct.error):
            return   # malformed response: discard silently

        with self._lock:
            entry = self._pending.get(response.query_id)

        if entry is not None:
            ev, container = entry
            container.append(response)
            ev.set()

    def resolve(
        self,
        name:    str,
        qtype:   int = DNS_TYPE_A,
        retries: int = DNS_MAX_RETRIES,
    ) -> List[str]:
        """
        Resolve name via custom_dns_ip:53.

        Steps:
            1. Obtain router MAC from ARP cache (must be populated by start()).
            2. Build DNS query → UDP → IPv4 → Ethernet frame.
            3. Register pending waiter, send frame, wait for reply.
            4. Parse response; follow CNAME if present.
            5. Return list of IP strings (A or AAAA records per qtype).

        Raises:
            ARPTimeoutError  — router MAC not in cache (start() not called?)
            DNSTimeoutError  — no response after retries × DNS_QUERY_TIMEOUT
            DNSError         — server returned non-zero RCODE
            PolicyViolation  — filter rejected the outbound DNS frame
        """
        router_mac = self._arp.get(str(self._cfg.router_ip))
        if router_mac is None:
            raise ARPTimeoutError(str(self._cfg.router_ip), 0.0)

        # Validate outbound policy before constructing any packets
        if not self._filter.check_outbound_udp(self._cfg.custom_dns_ip, DNS_PORT):
            raise PolicyViolation(
                'outbound',
                f'DNS query to {self._cfg.custom_dns_ip}:{DNS_PORT} blocked by filter',
            )

        src_port = self._allocate_src_port()
        last_exc: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            query_id  = random.randint(1, 0xFFFF)
            ip_ident  = self._allocate_ip_id()

            # ── Build DNS query payload ───────────────────────────────────────
            dns_bytes = DNSCodec.encode_query(query_id, name, qtype)

            # ── Wrap in UDP ───────────────────────────────────────────────────
            udp_bytes = UDPCodec.encode(UDPDatagram(
                src_port=src_port,
                dst_port=DNS_PORT,
                payload=dns_bytes,
            ))

            # ── Wrap in IPv4 ─────────────────────────────────────────────────
            ip_bytes = IPv4Codec.encode(IPv4Packet(
                src_ip=self._cfg.host_ip,
                dst_ip=self._cfg.custom_dns_ip,
                protocol=IPPROTO_UDP,
                payload=udp_bytes,
                ttl=64,
                identification=ip_ident,
                flags_offset=IP_FLAG_DF,
            ))

            # ── Wrap in Ethernet — dst is router MAC (Layer-2 gateway) ────────
            frame = EtherCodec.encode(
                dst=router_mac,
                src=self._iface.host_mac,
                ethertype=ETH_P_IP,
                payload=ip_bytes,
            )

            # ── Register waiter BEFORE send to eliminate race ─────────────────
            ev:        threading.Event   = threading.Event()
            container: List[DNSResponse] = []
            with self._lock:
                self._pending[query_id] = (ev, container)

            try:
                self._iface.send_frame(frame)
                got_reply = ev.wait(timeout=DNS_QUERY_TIMEOUT)
            finally:
                with self._lock:
                    self._pending.pop(query_id, None)

            if not got_reply:
                last_exc = DNSTimeoutError(name, DNS_QUERY_TIMEOUT)
                continue   # retry

            response: DNSResponse = container[0]

            # ── Extract records by requested type ─────────────────────────────
            if qtype == DNS_TYPE_A:
                addrs = response.a_records()
            elif qtype == DNS_TYPE_AAAA:
                addrs = response.aaaa_records()
            else:
                addrs = [r.value for r in response.answers if r.rtype == qtype]

            if addrs:
                return addrs

            # ── Follow CNAME if present ───────────────────────────────────────
            cnames = response.cname_records()
            if cnames:
                # Recursive resolution; retries budget is shared
                return self.resolve(cnames[0], qtype, retries - attempt)

            raise DNSError(
                f"No {qtype}-type records for {name!r} "
                f"(answer count: {len(response.answers)})",
                0,
            )

        # All retries exhausted
        raise last_exc  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ETHERSSENTINEL  (primary public interface)
# Orchestrates PhysicalInterface, ARPCache, PacketFilter, and DNSStub.
# Every public method is @agent_method decorated for AIOS kernel integration.
# ════════════════════════════════════════════════════════════════════════════════

class EtherSentinel:
    """
    EtherSentinel — Direct MAC-Layer Network Fabric for AIOS.

    Lifecycle:
        cfg = NetConfig(interface='eth0', host_ip='192.168.1.50', ...)
        sentinel = EtherSentinel(cfg)
        sentinel.start()          # opens socket, discovers router MAC, RX loop
        addrs = sentinel.resolve_dns("example.com")
        sentinel.stop()

    Context-manager protocol:
        with EtherSentinel(cfg) as sentinel:
            addrs = sentinel.resolve_dns("api.internal")

    Agent-callable methods:
        sentinel.start()          → bool
        sentinel.stop()           → None
        sentinel.resolve_dns()    → List[str]
        sentinel.arp_resolve()    → str
        sentinel.status()         → Dict[str, Any]
        sentinel.get_arp_table()  → Dict[str, str]
        sentinel.purge_arp()      → int

    All subsystems share a single NetConfig instance.  Mutations to config
    (router_mac and host_mac populated during start()) are visible to all
    subsystems immediately via the shared reference.
    """

    VERSION: Tuple[int, int, int] = (1, 0, 0)

    def __init__(self, config: NetConfig) -> None:
        self._cfg    = config
        self._iface  = PhysicalInterface(config)
        self._filter = PacketFilter(config)
        self._arp    = ARPCache()
        self._dns    = DNSStub(config, self._iface, self._arp, self._filter)
        self._running = False
        self._lock   = threading.RLock()

    # ── Internal frame handlers ───────────────────────────────────────────────

    def _handle_arp_frame(
        self,
        src_mac: MACAddress,
        dst_mac: MACAddress,
        payload: bytes,
    ) -> None:
        """
        Receive handler for ETH_P_ARP frames.
        Applies inbound filter, decodes, updates ARP cache on replies.
        Gratuitous ARP announcements update the cache unconditionally
        (sender_ip not 0.0.0.0).
        """
        if not self._filter.check_inbound_arp(src_mac):
            return

        try:
            pkt = ARPCodec.decode(payload)
        except ValueError:
            return

        # Always learn sender mapping (covers gratuitous ARP and replies)
        if pkt.sender_ip != IPv4Address('0.0.0.0'):
            self._arp.set(str(pkt.sender_ip), pkt.sender_mac)

    def _handle_ip_frame(
        self,
        src_mac: MACAddress,
        dst_mac: MACAddress,
        payload: bytes,
    ) -> None:
        """
        Receive handler for ETH_P_IP frames.
        Applies inbound filter per protocol/port, dispatches UDP payload
        to DNSStub when src_ip == custom_dns_ip and src_port == 53.
        """
        try:
            ip_pkt = IPv4Codec.decode(payload)
        except ValueError:
            return

        if ip_pkt.protocol == IPPROTO_UDP:
            try:
                udp = UDPCodec.decode(ip_pkt.payload)
            except ValueError:
                return

            if not self._filter.check_inbound_ip(
                ip_pkt.src_ip, IPPROTO_UDP, udp.src_port
            ):
                return

            if udp.src_port == DNS_PORT:
                self._dns.handle_udp_payload(
                    ip_pkt.src_ip,
                    udp.src_port,
                    udp.payload,
                )
        else:
            # Non-UDP IP traffic: apply filter for accounting, then discard.
            # EtherSentinel only processes DNS; TCP and ICMP are not supported.
            self._filter.check_inbound_ip(ip_pkt.src_ip, ip_pkt.protocol, None)

    def _send_arp_request(self, target_ip: IPv4Address) -> None:
        """
        Send an ARP REQUEST broadcast to discover target_ip's MAC.
        Raises PolicyViolation if the filter rejects the broadcast.
        Called internally during start() and arp_resolve().
        """
        if not self._filter.check_outbound_arp(MACAddress.BROADCAST):
            raise PolicyViolation(
                'outbound',
                f'ARP broadcast request for {target_ip} rejected by filter',
            )
        arp_bytes = ARPCodec.encode(ARPPacket(
            operation=ARP_OP_REQUEST,
            sender_mac=self._iface.host_mac,
            sender_ip=self._cfg.host_ip,
            target_mac=MACAddress.ZERO,
            target_ip=target_ip,
        ))
        frame = EtherCodec.encode(
            dst=MACAddress.BROADCAST,
            src=self._iface.host_mac,
            ethertype=ETH_P_ARP,
            payload=arp_bytes,
        )
        self._iface.send_frame(frame)

    # ── Public agent-method interface ─────────────────────────────────────────

    @agent_method(
        name="sentinel.start",
        description=(
            "Open the physical interface, auto-detect host MAC, discover "
            "router MAC via ARP if not supplied, and start the RX loop.  "
            "Must be called before any other method."
        ),
        parameters={},
        returns="bool",
        priority=AgentPriority.HIGH,
        owner="network",
    )
    def start(self) -> bool:
        """
        Initialise and start EtherSentinel.

        Steps:
            1. Open AF_PACKET socket → auto-detect host MAC via SIOCGIFHWADDR.
            2. Register ARP and IP frame handlers.
            3. Start EtherSentinel-RX daemon thread.
            4. If router_mac is already set in config, populate ARP cache.
               Otherwise, send ARP broadcast for router_ip and wait up to
               ARP_REQUEST_TIMEOUT seconds for a reply.
            5. Set self._running = True.

        Returns True on success.
        Raises InterfaceError, PrivilegeError, or ARPTimeoutError on failure.
        If this method raises, the interface is closed and resources cleaned up.
        """
        with self._lock:
            if self._running:
                return True   # idempotent

            try:
                # ── Step 1: Open socket, detect host MAC ──────────────────────
                self._iface.open()

                # ── Step 2: Register frame handlers ──────────────────────────
                self._iface.register_handler(ETH_P_ARP, self._handle_arp_frame)
                self._iface.register_handler(ETH_P_IP,  self._handle_ip_frame)

                # ── Step 3: Start RX thread ───────────────────────────────────
                self._iface.start_receive_loop()

                # ── Step 4: Populate ARP cache with router MAC ────────────────
                if self._cfg.router_mac is not None:
                    # Caller provided router MAC: seed the ARP cache directly
                    self._arp.set(str(self._cfg.router_ip), self._cfg.router_mac)
                else:
                    # Discover router MAC via ARP broadcast
                    self._send_arp_request(self._cfg.router_ip)
                    discovered = self._arp.wait_for(
                        str(self._cfg.router_ip),
                        timeout=ARP_REQUEST_TIMEOUT,
                    )
                    if discovered is None:
                        raise ARPTimeoutError(
                            str(self._cfg.router_ip),
                            ARP_REQUEST_TIMEOUT,
                        )
                    # Mutate shared config so PacketFilter sees the MAC too
                    self._cfg.router_mac = discovered

                # ── Step 5: Mark running ──────────────────────────────────────
                self._running = True
                return True

            except Exception:
                # Clean up on any failure during start
                try:
                    self._iface.close()
                except Exception:
                    pass
                raise

    @agent_method(
        name="sentinel.stop",
        description="Gracefully stop the RX loop and close the raw socket.",
        parameters={},
        returns="None",
        priority=AgentPriority.HIGH,
        owner="network",
    )
    def stop(self) -> None:
        """
        Halt EtherSentinel.  Safe to call if not running.
        Signals the RX thread to exit and closes the AF_PACKET socket.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._iface.close()

    @agent_method(
        name="sentinel.resolve_dns",
        description=(
            "Resolve a hostname using the configured custom DNS server "
            "(never the router's built-in DNS).  Returns a list of IP "
            "address strings.  Follows CNAME chains automatically."
        ),
        parameters={
            "name":    {"type": "str", "desc": "Fully qualified domain name"},
            "qtype":   {"type": "int", "desc": "DNS_TYPE_A=1 (default) or DNS_TYPE_AAAA=28"},
            "retries": {"type": "int", "desc": f"Retry count on timeout (default {DNS_MAX_RETRIES})"},
        },
        returns="List[str]",
        priority=AgentPriority.NORMAL,
        owner="network",
    )
    def resolve_dns(
        self,
        name:    str,
        qtype:   int = DNS_TYPE_A,
        retries: int = DNS_MAX_RETRIES,
    ) -> List[str]:
        """
        Resolve name via custom_dns_ip:53.

        Raises EtherSentinelError subclasses (never standard socket errors).
        The caller should catch DNSError, DNSTimeoutError, ARPTimeoutError.
        """
        if not self._running:
            raise InterfaceError(
                "EtherSentinel is not running.  Call start() before resolve_dns()."
            )
        return self._dns.resolve(name, qtype, retries)

    @agent_method(
        name="sentinel.arp_resolve",
        description=(
            "Resolve an IPv4 address to a MAC address via ARP.  "
            "Returns a colon-separated MAC string.  "
            "Uses cache if available; sends ARP request otherwise."
        ),
        parameters={
            "ip":    {"type": "str",  "desc": "IPv4 address to resolve (dotted decimal)"},
            "force": {"type": "bool", "desc": "Re-send ARP even if cache hit (default: False)"},
        },
        returns="str",
        priority=AgentPriority.NORMAL,
        owner="network",
    )
    def arp_resolve(self, ip: str, force: bool = False) -> str:
        """
        Resolve IP → MAC via ARP.

        If force=False and the IP is in the live ARP cache, return immediately.
        Otherwise, invalidate any existing entry, broadcast an ARP request,
        and wait up to ARP_REQUEST_TIMEOUT seconds.
        Raises ARPTimeoutError if no reply arrives.
        """
        if not self._running:
            raise InterfaceError(
                "EtherSentinel is not running.  Call start() before arp_resolve()."
            )
        target = IPv4Address(ip)

        if not force:
            cached = self._arp.get(ip)
            if cached is not None:
                return str(cached)

        self._arp.invalidate(ip)
        self._send_arp_request(target)
        mac = self._arp.wait_for(ip, timeout=ARP_REQUEST_TIMEOUT)

        if mac is None:
            raise ARPTimeoutError(ip, ARP_REQUEST_TIMEOUT)

        return str(mac)

    @agent_method(
        name="sentinel.status",
        description=(
            "Return a complete operational snapshot: running state, "
            "interface, address configuration, ARP table, filter stats, "
            "and interface I/O counters."
        ),
        parameters={},
        returns="Dict[str, Any]",
        priority=AgentPriority.LOW,
        owner="network",
    )
    def status(self) -> Dict[str, Any]:
        """Full operational status of the EtherSentinel fabric."""
        return {
            'version':      '{}.{}.{}'.format(*self.VERSION),
            'running':      self._running,
            'aios_kernel':  _AIOS_INTEGRATED,
            'interface':    self._cfg.interface,
            'host_ip':      str(self._cfg.host_ip),
            'host_mac':     str(self._iface.host_mac) if self._iface.host_mac else 'undetected',
            'router_ip':    str(self._cfg.router_ip),
            'router_mac':   str(self._cfg.router_mac) if self._cfg.router_mac else 'pending',
            'dns_server':   str(self._cfg.custom_dns_ip),
            'subnet_mask':  str(self._cfg.subnet_mask),
            'arp_entries':  len(self._arp),
            'arp_table':    self._arp.as_dict(),
            'filter_stats': self._filter.stats,
            'iface_stats':  self._iface.stats,
        }

    @agent_method(
        name="sentinel.get_arp_table",
        description="Return the current live ARP cache as {ip_str: mac_str}.",
        parameters={},
        returns="Dict[str, str]",
        priority=AgentPriority.LOW,
        owner="network",
    )
    def get_arp_table(self) -> Dict[str, str]:
        return self._arp.as_dict()

    @agent_method(
        name="sentinel.purge_arp",
        description="Evict all TTL-expired entries from the ARP cache.",
        parameters={},
        returns="int",
        priority=AgentPriority.LOW,
        owner="network",
    )
    def purge_arp(self) -> int:
        """Remove expired ARP entries.  Returns count removed."""
        return self._arp.purge_expired()

    # ── Context-manager protocol ──────────────────────────────────────────────

    def __enter__(self) -> 'EtherSentinel':
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"EtherSentinel("
            f"iface={self._cfg.interface!r}, "
            f"host={self._cfg.host_ip}, "
            f"router={self._cfg.router_ip}, "
            f"dns={self._cfg.custom_dns_ip}, "
            f"running={self._running})"
        )


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DEMONSTRATION ENTRY POINT
# Edit NetConfig fields below for your network before running.
# Requires: Linux, root or CAP_NET_RAW, a live Ethernet interface.
# ════════════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    """
    Demonstrate EtherSentinel operation.

    Configured values below are examples only — replace with your actual
    network parameters before running.  The test:
        1. Opens the interface and discovers the router MAC via ARP.
        2. Resolves 'example.com' using the custom DNS server (1.1.1.1).
        3. Prints the ARP table and filter/interface statistics.
        4. Stops cleanly.
    """
    BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   EtherSentinel — Direct MAC Network Fabric                  ║
║   AIOS Network Subsystem v{v}                            ║
╚══════════════════════════════════════════════════════════════╝
""".format(v='.'.join(str(x) for x in EtherSentinel.VERSION))
    print(BANNER)

    # ── Configure for your network ────────────────────────────────────────────
    cfg = NetConfig(
        interface     = 'eth0',          # replace with your interface name
        host_ip       = '192.168.1.100', # your static IP
        router_ip     = '192.168.1.1',   # your gateway/router IP
        router_mac    = None,            # None: EtherSentinel discovers via ARP
        custom_dns_ip = '1.1.1.1',       # Cloudflare — NOT the router's DNS
        subnet_mask   = '255.255.255.0',
        host_mac      = None,            # None: auto-detected via SIOCGIFHWADDR
    )

    print(f"  Interface  : {cfg.interface}")
    print(f"  Host IP    : {cfg.host_ip}")
    print(f"  Router     : {cfg.router_ip}")
    print(f"  DNS server : {cfg.custom_dns_ip}  (non-router)")
    print(f"  Subnet     : {cfg.subnet_mask}")
    print()

    sentinel = EtherSentinel(cfg)

    try:
        print("[+] Opening AF_PACKET socket and starting receive loop ...")
        sentinel.start()
        print(f"[+] Interface open.  Host MAC : {sentinel._iface.host_mac}")
        print(f"[+] Router MAC       : {cfg.router_mac}  (discovered via ARP)")
        print()

        print("[+] Resolving example.com via custom DNS ...")
        a_records = sentinel.resolve_dns("example.com", qtype=DNS_TYPE_A)
        print(f"    example.com  A  → {a_records}")

        aaaa_records = sentinel.resolve_dns("example.com", qtype=DNS_TYPE_AAAA)
        print(f"    example.com  AAAA → {aaaa_records}")
        print()

        status = sentinel.status()
        print("[+] ARP table:")
        for ip, mac in status['arp_table'].items():
            print(f"    {ip:<20} → {mac}")

        print()
        print("[+] PacketFilter statistics:")
        for k, v in status['filter_stats'].items():
            print(f"    {k:<26}: {v}")

        print()
        print("[+] Interface I/O statistics:")
        for k, v in sorted(status['iface_stats'].items()):
            print(f"    {k:<26}: {v}")

    except PrivilegeError as exc:
        print(f"\n[!] Privilege error:\n    {exc}")
    except ARPTimeoutError as exc:
        print(f"\n[!] {exc}")
        print("    Verify router_ip is correct and reachable on this interface.")
    except DNSTimeoutError as exc:
        print(f"\n[!] {exc}")
        print("    Verify custom_dns_ip is reachable through the router.")
    except DNSError as exc:
        print(f"\n[!] DNS error (RCODE={exc.rcode}): {exc}")
    except InterfaceError as exc:
        print(f"\n[!] Interface error: {exc}")
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    finally:
        print("\n[+] Stopping EtherSentinel ...")
        sentinel.stop()
        print("[+] Stopped.  Socket closed.")


if __name__ == '__main__':
    _demo()
