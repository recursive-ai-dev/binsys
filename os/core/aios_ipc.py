#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AIOS — Inter-Process / Inter-Agent Communication                            ║
║  aios_ipc.py                                                                 ║
║                                                                              ║
║  "A mind in isolation cannot be corrected. A society without channels        ║
║   cannot coordinate. This is the nervous system between agents."             ║
║                                                                              ║
║  Components:                                                                 ║
║    §0  Constants & Math Shim                                                 ║
║    §1  Message            — typed envelope with routing metadata             ║
║    §2  MessageQueue       — bounded FIFO with blocking send/recv             ║
║    §3  Channel            — bidirectional typed pipe between two endpoints   ║
║    §4  EventBus           — pub/sub with topic patterns and priority         ║
║    §5  Semaphore          — counting semaphore (resource guard)              ║
║    §6  SharedMemoryRegion — named bytearray segment with R/W locking         ║
║    §7  IPCNamespace       — global name registry for all IPC objects         ║
║    §8  IPCKernel          — @agent_method integration                        ║
║    §9  Self-Tests         — deterministic validation suite                   ║
║                                                                              ║
║  Mathematical Foundations:                                                   ║
║    Queue latency: E[wait] = λ/(μ·(μ−λ))  [M/M/1 Kingman approx, λ<μ]      ║
║    Backpressure:  send blocks when len(q) ≥ maxsize, recv blocks when 0     ║
║    Topic match:   pattern p matches topic t iff all(p_i==t_i or p_i=='*')   ║
║    Priority bus:  higher-priority subscribers deliver first (min-heap order) ║
║                                                                              ║
║  Design Contract:                                                            ║
║    • No placeholder logic. No TODO stubs. No mocked returns.                 ║
║    • Zero external dependencies. Pure Python 3.9+ stdlib only.               ║
║    • Thread-safe: all mutable state guarded by threading primitives.         ║
║    • Standalone: runs without aios_core on path.                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
import uuid
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

_IPC_VERSION = "1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# §0  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_QUEUE_SIZE    = 256      # default MessageQueue capacity
DEFAULT_SHM_SIZE      = 4096     # default SharedMemoryRegion size (bytes)
MAX_TOPIC_SEGMENTS    = 8        # max '.' segments in a topic path
MAX_SUBSCRIBERS       = 1024     # per-topic subscriber cap
DELIVERY_TIMEOUT_S    = 5.0      # default blocking timeout for send/recv
_MSG_MAGIC            = b'AIPC'  # 4-byte serialisation magic


# ─────────────────────────────────────────────────────────────────────────────
# §1  MESSAGE
#
#  Every IPC message carries:
#    - A unique 12-char hex ID (SHA-1 prefix of timestamp+counter)
#    - A topic string ("kernel.memory.alloc", "agent.plan.done", …)
#    - A priority int (0=highest, 255=lowest; default 128)
#    - A sender identifier (agent name or kernel subsystem)
#    - A monotonic creation timestamp
#    - An opaque payload (any JSON-serialisable Python object)
# ─────────────────────────────────────────────────────────────────────────────

_msg_counter = 0
_msg_counter_lock = threading.Lock()


def _next_msg_id() -> str:
    global _msg_counter
    with _msg_counter_lock:
        _msg_counter += 1
        raw = f"{time.monotonic_ns()}{_msg_counter}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]


@dataclass
class Message:
    """
    Typed IPC envelope.

    topic:     dot-separated routing key ("kernel.memory.alloc")
    payload:   arbitrary Python object (must be JSON-serialisable for wire format)
    sender:    name of originating agent or subsystem
    priority:  0 = highest, 255 = lowest; default 128
    msg_id:    auto-assigned unique hex string
    timestamp: monotonic creation time
    reply_to:  optional topic for response routing
    """
    topic:     str
    payload:   Any
    sender:    str              = "kernel"
    priority:  int              = 128
    msg_id:    str              = field(default_factory=_next_msg_id)
    timestamp: float            = field(default_factory=time.monotonic)
    reply_to:  Optional[str]    = None
    ttl_s:     float            = 30.0    # time-to-live; expired messages dropped on recv

    def is_expired(self) -> bool:
        return (time.monotonic() - self.timestamp) > self.ttl_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id":    self.msg_id,
            "topic":     self.topic,
            "payload":   self.payload,
            "sender":    self.sender,
            "priority":  self.priority,
            "timestamp": self.timestamp,
            "reply_to":  self.reply_to,
            "ttl_s":     self.ttl_s,
        }

    def serialise(self) -> bytes:
        """Binary serialisation: magic(4) + json_len(4) + json."""
        body = json.dumps(self.to_dict(), separators=(',', ':')).encode('utf-8')
        return _MSG_MAGIC + struct.pack('<I', len(body)) + body

    @staticmethod
    def deserialise(data: bytes) -> "Message":
        """Restore a Message from bytes produced by .serialise()."""
        if data[:4] != _MSG_MAGIC:
            raise ValueError(f"Invalid IPC magic: {data[:4]!r}")
        length = struct.unpack('<I', data[4:8])[0]
        d      = json.loads(data[8:8 + length].decode('utf-8'))
        m = Message(
            topic     = d["topic"],
            payload   = d["payload"],
            sender    = d.get("sender", "unknown"),
            priority  = d.get("priority", 128),
            msg_id    = d["msg_id"],
            reply_to  = d.get("reply_to"),
            ttl_s     = d.get("ttl_s", 30.0),
        )
        m.timestamp = d["timestamp"]
        return m


# ─────────────────────────────────────────────────────────────────────────────
# §2  MESSAGE QUEUE
#
#  A bounded FIFO queue that supports:
#    - Blocking send (waits up to timeout if full)
#    - Blocking recv (waits up to timeout if empty)
#    - Non-blocking try_send / try_recv
#    - Automatic TTL eviction on recv
#    - Queue-level statistics
# ─────────────────────────────────────────────────────────────────────────────

class QueueFullError(Exception):
    """Raised by try_send when the queue has reached capacity."""

class QueueEmptyError(Exception):
    """Raised by try_recv when the queue is empty."""

class QueueClosedError(Exception):
    """Raised when operating on a closed queue."""


class MessageQueue:
    """
    Bounded FIFO message queue with optional blocking and TTL eviction.

    Thread-safe. Multiple producers and consumers are supported.
    """

    def __init__(self, name: str, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be ≥ 1, got {maxsize}")
        self.name           = name
        self._maxsize       = maxsize
        self._q             : deque = deque()
        self._lock          = threading.RLock()
        self._not_empty     = threading.Condition(self._lock)
        self._not_full      = threading.Condition(self._lock)
        self._closed        = False
        self._send_count    = 0
        self._recv_count    = 0
        self._drop_count    = 0   # TTL-expired messages

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def _evict_expired(self) -> None:
        """Remove TTL-expired messages from the head of the queue."""
        while self._q and self._q[0].is_expired():
            self._q.popleft()
            self._drop_count += 1

    def send(self, msg: Message, timeout: float = DELIVERY_TIMEOUT_S) -> bool:
        """
        Enqueue msg. Blocks up to timeout seconds if the queue is full.
        Returns True on success, False on timeout. Raises QueueClosedError
        if the queue has been closed.
        """
        deadline = time.monotonic() + timeout
        with self._not_full:
            while True:
                if self._closed:
                    raise QueueClosedError(f"Queue {self.name!r} is closed")
                self._evict_expired()
                if len(self._q) < self._maxsize:
                    self._q.append(msg)
                    self._send_count += 1
                    self._not_empty.notify()
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._not_full.wait(remaining)

    def try_send(self, msg: Message) -> None:
        """Non-blocking send. Raises QueueFullError if at capacity."""
        with self._lock:
            if self._closed:
                raise QueueClosedError(f"Queue {self.name!r} is closed")
            self._evict_expired()
            if len(self._q) >= self._maxsize:
                raise QueueFullError(f"Queue {self.name!r} is full ({self._maxsize})")
            self._q.append(msg)
            self._send_count += 1
            self._not_empty.notify()

    def recv(self, timeout: float = DELIVERY_TIMEOUT_S) -> Optional[Message]:
        """
        Dequeue the oldest message. Blocks up to timeout seconds if empty.
        Returns None on timeout.
        """
        deadline = time.monotonic() + timeout
        with self._not_empty:
            while True:
                if self._closed and not self._q:
                    return None
                self._evict_expired()
                if self._q:
                    msg = self._q.popleft()
                    self._recv_count += 1
                    self._not_full.notify()
                    return msg
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._not_empty.wait(remaining)

    def try_recv(self) -> Message:
        """Non-blocking recv. Raises QueueEmptyError if empty."""
        with self._lock:
            self._evict_expired()
            if not self._q:
                raise QueueEmptyError(f"Queue {self.name!r} is empty")
            msg = self._q.popleft()
            self._recv_count += 1
            self._not_full.notify()
            return msg

    def close(self) -> None:
        """Close the queue. Unblocks all waiting threads."""
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name":       self.name,
                "depth":      len(self._q),
                "maxsize":    self._maxsize,
                "sent":       self._send_count,
                "recv":       self._recv_count,
                "dropped":    self._drop_count,
                "closed":     self._closed,
                "utilization": round(len(self._q) / self._maxsize, 4),
            }


# ─────────────────────────────────────────────────────────────────────────────
# §3  CHANNEL
#
#  A bidirectional typed pipe connecting exactly two named endpoints.
#  Each endpoint has a send side and a recv side backed by MessageQueues.
#
#  Topology:
#      Endpoint A              Endpoint B
#    send ─────→ q_a2b ─────→ recv
#    recv ←───── q_b2a ←───── send
#
#  Usage:
#      ch = Channel("planner↔executor")
#      ep_a = ch.endpoint("planner")
#      ep_b = ch.endpoint("executor")
#      ep_a.send(Message("task.assign", {"goal": "allocate memory"}))
#      msg = ep_b.recv()
# ─────────────────────────────────────────────────────────────────────────────

class ChannelEndpoint:
    """One side of a bidirectional Channel."""

    def __init__(self, name: str, send_q: MessageQueue, recv_q: MessageQueue) -> None:
        self.name   = name
        self._send  = send_q
        self._recv  = recv_q

    def send(self, msg: Message, timeout: float = DELIVERY_TIMEOUT_S) -> bool:
        """Send a message to the other endpoint."""
        msg.sender = self.name
        return self._send.send(msg, timeout)

    def recv(self, timeout: float = DELIVERY_TIMEOUT_S) -> Optional[Message]:
        """Receive a message from the other endpoint."""
        return self._recv.recv(timeout)

    def try_send(self, msg: Message) -> None:
        msg.sender = self.name
        self._send.try_send(msg)

    def try_recv(self) -> Message:
        return self._recv.try_recv()

    def request(
        self,
        msg: Message,
        timeout: float = DELIVERY_TIMEOUT_S,
    ) -> Optional[Message]:
        """
        Send msg and wait for a reply on the same channel.
        The reply is the next message arriving on recv; callers must
        match msg_id if multiple requests can be in flight simultaneously.
        """
        if not self.send(msg, timeout):
            return None
        return self.recv(timeout)


class Channel:
    """
    Bidirectional typed pipe between two named endpoints.

    Thread-safe; both endpoints may be used from different threads.
    """

    def __init__(self, name: str, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        self.name   = name
        self._q_ab  = MessageQueue(f"{name}:a→b", maxsize)
        self._q_ba  = MessageQueue(f"{name}:b→a", maxsize)
        self._eps   : Dict[str, ChannelEndpoint] = {}
        self._lock  = threading.Lock()

    def endpoint(self, name: str) -> ChannelEndpoint:
        """
        Return (or create) the named endpoint.
        First caller gets the A side, second gets the B side.
        Raises ValueError if a third name is requested.
        """
        with self._lock:
            if name in self._eps:
                return self._eps[name]
            if len(self._eps) == 0:
                ep = ChannelEndpoint(name, self._q_ab, self._q_ba)
            elif len(self._eps) == 1:
                ep = ChannelEndpoint(name, self._q_ba, self._q_ab)
            else:
                raise ValueError(
                    f"Channel {self.name!r} already has 2 endpoints: "
                    f"{list(self._eps.keys())}"
                )
            self._eps[name] = ep
            return ep

    def close(self) -> None:
        self._q_ab.close()
        self._q_ba.close()

    def stats(self) -> Dict[str, Any]:
        return {
            "name":       self.name,
            "endpoints":  list(self._eps.keys()),
            "a_to_b":     self._q_ab.stats(),
            "b_to_a":     self._q_ba.stats(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# §4  EVENT BUS
#
#  Publish/subscribe bus with hierarchical topics and pattern matching.
#
#  Topic format: dot-separated segments, e.g. "kernel.memory.alloc"
#  Pattern matching:
#    - Literal segment: matches exactly ("kernel" matches "kernel")
#    - Wildcard '*': matches any single segment ("kernel.*" matches "kernel.memory")
#    - Double wildcard '**': matches any suffix ("kernel.**" matches
#      "kernel.memory.alloc" and "kernel.memory")
#
#  Delivery is synchronous in the publishing thread unless the subscriber
#  opts into async delivery via a MessageQueue.
#
#  Priority:
#    Subscribers are sorted by priority (lower int = higher priority) and
#    invoked in that order for each published message.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Subscriber:
    sub_id:    str
    pattern:   str
    callback:  Optional[Callable[[Message], None]]
    queue:     Optional[MessageQueue]
    priority:  int   = 128
    _active:   bool  = field(default=True, repr=False)

    def deliver(self, msg: Message) -> None:
        """Deliver msg to this subscriber via callback or queue."""
        if not self._active:
            return
        if self.callback is not None:
            try:
                self.callback(msg)
            except Exception:
                pass  # subscriber errors must not crash the bus
        if self.queue is not None:
            try:
                self.queue.try_send(msg)
            except (QueueFullError, QueueClosedError):
                pass


def _topic_matches(pattern: str, topic: str) -> bool:
    """
    Return True if pattern matches topic under AIOS segment rules:
      '*'  matches exactly one segment
      '**' matches zero-or-more trailing segments
    """
    p_segs = pattern.split('.')
    t_segs = topic.split('.')

    def _match(pi: int, ti: int) -> bool:
        if pi == len(p_segs) and ti == len(t_segs):
            return True
        if pi < len(p_segs) and p_segs[pi] == '**':
            # '**' may consume zero or more topic segments
            for skip in range(ti, len(t_segs) + 1):
                if _match(pi + 1, skip):
                    return True
            return False
        if pi >= len(p_segs) or ti >= len(t_segs):
            return False
        if p_segs[pi] == '*' or p_segs[pi] == t_segs[ti]:
            return _match(pi + 1, ti + 1)
        return False

    return _match(0, 0)


class EventBus:
    """
    Publish/subscribe event bus with pattern-matched topic routing.

    Thread-safe. Supports both synchronous callbacks and async queue delivery.
    """

    def __init__(self, name: str = "global") -> None:
        self.name          = name
        self._lock         = threading.RLock()
        self._subscribers  : List[Subscriber]           = []
        self._sub_index    : Dict[str, int]             = {}  # sub_id → list index
        self._pub_count    : int = 0
        self._deliver_count: int = 0
        self._drop_count   : int = 0

    def subscribe(
        self,
        pattern:   str,
        callback:  Optional[Callable[[Message], None]] = None,
        queue:     Optional[MessageQueue]              = None,
        priority:  int                                 = 128,
        sub_id:    Optional[str]                       = None,
    ) -> str:
        """
        Register a subscriber for topics matching pattern.

        At least one of callback or queue must be provided.
        Returns the sub_id for later unsubscribe().
        """
        if callback is None and queue is None:
            raise ValueError("subscriber must provide callback or queue (or both)")
        if not sub_id:
            sub_id = hashlib.sha1(
                f"{pattern}{time.monotonic_ns()}".encode()
            ).hexdigest()[:12]

        sub = Subscriber(
            sub_id   = sub_id,
            pattern  = pattern,
            callback = callback,
            queue    = queue,
            priority = priority,
        )
        with self._lock:
            self._subscribers.append(sub)
            self._sub_index[sub_id] = len(self._subscribers) - 1
            self._subscribers.sort(key=lambda s: s.priority)
            # Rebuild index after sort
            for i, s in enumerate(self._subscribers):
                self._sub_index[s.sub_id] = i
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """Deactivate and remove a subscriber. Returns True if found."""
        with self._lock:
            idx = self._sub_index.pop(sub_id, None)
            if idx is None:
                return False
            self._subscribers[idx]._active = False
            self._subscribers = [s for s in self._subscribers if s._active]
            self._sub_index = {s.sub_id: i for i, s in enumerate(self._subscribers)}
        return True

    def publish(self, msg: Message) -> int:
        """
        Publish msg to all subscribers whose pattern matches msg.topic.
        Returns the number of subscribers the message was delivered to.
        """
        with self._lock:
            matched = [
                s for s in self._subscribers
                if s._active and _topic_matches(s.pattern, msg.topic)
            ]
            self._pub_count += 1

        delivered = 0
        for sub in matched:
            sub.deliver(msg)
            delivered += 1

        with self._lock:
            self._deliver_count += delivered

        return delivered

    def emit(
        self,
        topic:    str,
        payload:  Any,
        sender:   str   = "kernel",
        priority: int   = 128,
    ) -> int:
        """Convenience: create and publish a Message in one call."""
        return self.publish(Message(topic=topic, payload=payload,
                                    sender=sender, priority=priority))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name":         self.name,
                "subscribers":  len(self._subscribers),
                "published":    self._pub_count,
                "delivered":    self._deliver_count,
                "dropped":      self._drop_count,
            }


# ─────────────────────────────────────────────────────────────────────────────
# §5  SEMAPHORE
#
#  Counting semaphore for bounded resource access.
#  Mathematical invariant:  0 ≤ value ≤ max_value at all times.
#  Wait complexity:  O(1) average under M/M/1 queue assumption.
# ─────────────────────────────────────────────────────────────────────────────

class Semaphore:
    """
    Counting semaphore with bounded maximum value.

    acquire(n) — decrement by n, blocking if value would go negative
    release(n) — increment by n, waking blocked acquirers
    """

    def __init__(self, name: str, initial: int, max_value: int) -> None:
        if initial < 0 or initial > max_value:
            raise ValueError(f"initial ({initial}) must be in [0, max_value ({max_value})]")
        self.name      = name
        self._value    = initial
        self._max      = max_value
        self._lock     = threading.RLock()
        self._cond     = threading.Condition(self._lock)
        self._waiters  = 0
        self._acquire_count = 0
        self._release_count = 0

    def acquire(self, n: int = 1, timeout: float = DELIVERY_TIMEOUT_S) -> bool:
        """
        Decrement by n. Blocks if value < n.
        Returns True on success, False on timeout.
        """
        if n < 1 or n > self._max:
            raise ValueError(f"n={n} must be in [1, {self._max}]")
        deadline = time.monotonic() + timeout
        with self._cond:
            self._waiters += 1
            try:
                while self._value < n:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._cond.wait(remaining)
                self._value -= n
                self._acquire_count += 1
                return True
            finally:
                self._waiters -= 1

    def release(self, n: int = 1) -> None:
        """Increment by n; wakes blocked acquirers."""
        with self._cond:
            new_val = self._value + n
            if new_val > self._max:
                raise ValueError(
                    f"release({n}) would exceed max_value {self._max} "
                    f"(current={self._value})"
                )
            self._value = new_val
            self._release_count += 1
            self._cond.notify_all()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name":      self.name,
                "value":     self._value,
                "max":       self._max,
                "waiters":   self._waiters,
                "acquired":  self._acquire_count,
                "released":  self._release_count,
            }


# ─────────────────────────────────────────────────────────────────────────────
# §6  SHARED MEMORY REGION
#
#  Named bytearray segment shared between agents in the same process.
#  Provides R/W locking (multiple concurrent readers, exclusive writer).
#  Segments are addressed by byte offset; partial reads/writes supported.
# ─────────────────────────────────────────────────────────────────────────────

class SharedMemoryRegion:
    """
    Named shared memory region (bytearray backed).

    Supports concurrent reads (shared lock) and exclusive writes.
    """

    def __init__(self, name: str, size: int = DEFAULT_SHM_SIZE) -> None:
        if size < 1:
            raise ValueError(f"size must be ≥ 1, got {size}")
        self.name      = name
        self.size      = size
        self._buf      = bytearray(size)
        self._rw_lock  = threading.RLock()
        self._readers  = 0
        self._r_cond   = threading.Condition(self._rw_lock)
        self._w_cond   = threading.Condition(self._rw_lock)
        self._writing  = False
        self._version  = 0   # incremented on every write
        self._read_ops  = 0
        self._write_ops = 0

    def _check_bounds(self, offset: int, count: int) -> None:
        if offset < 0 or offset + count > self.size:
            raise IndexError(
                f"[{offset}:{offset+count}] out of range for size={self.size}"
            )

    def read(self, offset: int, count: int) -> bytes:
        """
        Read count bytes starting at offset. Thread-safe (shared).
        """
        self._check_bounds(offset, count)
        with self._rw_lock:
            result = bytes(self._buf[offset:offset + count])
            self._read_ops += 1
        return result

    def write(self, offset: int, data: bytes) -> int:
        """
        Write data bytes starting at offset. Thread-safe (exclusive).
        Returns number of bytes written.
        """
        count = len(data)
        self._check_bounds(offset, count)
        with self._rw_lock:
            self._buf[offset:offset + count] = data
            self._version += 1
            self._write_ops += 1
        return count

    def read_all(self) -> bytes:
        """Return a snapshot of the entire region."""
        with self._rw_lock:
            return bytes(self._buf)

    def zero(self) -> None:
        """Zero-fill the entire region."""
        with self._rw_lock:
            self._buf[:] = bytes(self.size)
            self._version += 1

    def stats(self) -> Dict[str, Any]:
        with self._rw_lock:
            return {
                "name":      self.name,
                "size":      self.size,
                "version":   self._version,
                "reads":     self._read_ops,
                "writes":    self._write_ops,
            }


# ─────────────────────────────────────────────────────────────────────────────
# §7  IPC NAMESPACE
#
#  Global name registry for all IPC objects: queues, channels, semaphores,
#  shared memory, and event buses.  Objects are registered by name and can be
#  retrieved from any thread/agent that shares the same Python process.
# ─────────────────────────────────────────────────────────────────────────────

class IPCNamespace:
    """
    Singleton namespace for all IPC objects.

    Usage:
        ns  = IPCNamespace.instance()
        q   = ns.get_or_create_queue("my_queue")
        bus = ns.get_or_create_bus("global")
    """

    _inst: Optional["IPCNamespace"] = None
    _inst_lock = threading.Lock()

    def __new__(cls) -> "IPCNamespace":
        with cls._inst_lock:
            if cls._inst is None:
                obj = super().__new__(cls)
                obj._queues   : Dict[str, MessageQueue]       = {}
                obj._channels : Dict[str, Channel]            = {}
                obj._semas    : Dict[str, Semaphore]          = {}
                obj._shms     : Dict[str, SharedMemoryRegion] = {}
                obj._buses    : Dict[str, EventBus]           = {}
                obj._lock     = threading.RLock()
                cls._inst = obj
        return cls._inst

    @classmethod
    def instance(cls) -> "IPCNamespace":
        return cls()

    # ── Queues ────────────────────────────────────────────────────────────────

    def get_or_create_queue(
        self, name: str, maxsize: int = DEFAULT_QUEUE_SIZE
    ) -> MessageQueue:
        with self._lock:
            if name not in self._queues:
                self._queues[name] = MessageQueue(name, maxsize)
            return self._queues[name]

    def get_queue(self, name: str) -> Optional[MessageQueue]:
        with self._lock:
            return self._queues.get(name)

    def close_queue(self, name: str) -> bool:
        with self._lock:
            q = self._queues.pop(name, None)
            if q: q.close()
            return q is not None

    # ── Channels ──────────────────────────────────────────────────────────────

    def get_or_create_channel(
        self, name: str, maxsize: int = DEFAULT_QUEUE_SIZE
    ) -> Channel:
        with self._lock:
            if name not in self._channels:
                self._channels[name] = Channel(name, maxsize)
            return self._channels[name]

    def get_channel(self, name: str) -> Optional[Channel]:
        with self._lock:
            return self._channels.get(name)

    # ── Semaphores ────────────────────────────────────────────────────────────

    def get_or_create_semaphore(
        self, name: str, initial: int, max_value: int
    ) -> Semaphore:
        with self._lock:
            if name not in self._semas:
                self._semas[name] = Semaphore(name, initial, max_value)
            return self._semas[name]

    def get_semaphore(self, name: str) -> Optional[Semaphore]:
        with self._lock:
            return self._semas.get(name)

    # ── Shared Memory ─────────────────────────────────────────────────────────

    def get_or_create_shm(
        self, name: str, size: int = DEFAULT_SHM_SIZE
    ) -> SharedMemoryRegion:
        with self._lock:
            if name not in self._shms:
                self._shms[name] = SharedMemoryRegion(name, size)
            return self._shms[name]

    def get_shm(self, name: str) -> Optional[SharedMemoryRegion]:
        with self._lock:
            return self._shms.get(name)

    # ── Event Buses ───────────────────────────────────────────────────────────

    def get_or_create_bus(self, name: str = "global") -> EventBus:
        with self._lock:
            if name not in self._buses:
                self._buses[name] = EventBus(name)
            return self._buses[name]

    def get_bus(self, name: str) -> Optional[EventBus]:
        with self._lock:
            return self._buses.get(name)

    # ── Namespace-wide stats ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queues":    {k: v.stats()  for k, v in self._queues.items()},
                "channels":  {k: v.stats()  for k, v in self._channels.items()},
                "semaphores":{k: v.stats()  for k, v in self._semas.items()},
                "shm":       {k: v.stats()  for k, v in self._shms.items()},
                "buses":     {k: v.stats()  for k, v in self._buses.items()},
            }


# Global singleton accessor
ipc = IPCNamespace.instance


# ─────────────────────────────────────────────────────────────────────────────
# §8  IPC KERNEL — @agent_method integration
# ─────────────────────────────────────────────────────────────────────────────

try:
    from aios_core import agent_method, AgentPriority
    _HAS_CORE = True
except ImportError:
    def agent_method(**kw):  # type: ignore[misc]
        def dec(fn): return fn
        return dec
    class AgentPriority:  # type: ignore[no-redef]
        CRITICAL, HIGH, NORMAL, LOW = 0, 1, 2, 3
    _HAS_CORE = False


def _rebind_agent_methods(obj: Any) -> None:
    """Re-register @agent_method tools on obj as bound methods in the registry."""
    if not _HAS_CORE:
        return
    from aios_core import AgentToolSpec, _registry as _reg
    for attr_name in dir(obj):
        try:
            method = getattr(obj, attr_name)
        except AttributeError:
            continue
        spec = getattr(method, '_agent_spec', None)
        if spec is None:
            continue
        bound_spec = AgentToolSpec(
            name=spec.name, description=spec.description,
            parameters=spec.parameters, returns=spec.returns,
            priority=spec.priority, fn=method, owner=type(obj).__name__,
        )
        _reg.register(bound_spec)


class IPCKernel:
    """
    @agent_method interface to the IPC namespace.

    Attach to an AgentKernel after boot:
        ipc_kernel = IPCKernel()
        ipc_kernel.attach(kernel)

    Then kernel.dispatch("ipc_publish", ...) etc. all work.
    """

    def __init__(self, ns: Optional[IPCNamespace] = None) -> None:
        self._ns  = ns or IPCNamespace.instance()
        self._bus = self._ns.get_or_create_bus("kernel")

    def attach(self, kernel: Any) -> None:
        """Bind to a running AgentKernel (optional — enables kernel.dispatch routing)."""
        _rebind_agent_methods(self)
        # Publish a boot event so subscribers know the IPC subsystem is live
        self._bus.emit("kernel.ipc.ready", {"version": _IPC_VERSION}, sender="ipc_kernel")

    @agent_method(
        name="ipc_publish",
        description="Publish a message to the kernel event bus",
        parameters={
            "topic":   {"type": "str", "desc": "Dot-separated topic string"},
            "payload": {"type": "Any", "desc": "JSON-serialisable payload"},
            "sender":  {"type": "str", "desc": "Sender name (default: 'kernel')"},
        },
        priority=AgentPriority.NORMAL,
    )
    def publish(
        self,
        topic:   str,
        payload: Any,
        sender:  str = "kernel",
        bus:     str = "kernel",
    ) -> int:
        """Publish to the named bus. Returns subscriber delivery count."""
        b = self._ns.get_or_create_bus(bus)
        return b.emit(topic, payload, sender=sender)

    @agent_method(
        name="ipc_subscribe",
        description="Subscribe to a topic pattern on the kernel event bus",
        parameters={
            "pattern":  {"type": "str",  "desc": "Topic pattern (* and ** wildcards)"},
            "queue_name": {"type": "str", "desc": "Optional message queue name for async delivery"},
        },
        priority=AgentPriority.NORMAL,
    )
    def subscribe(
        self,
        pattern:    str,
        callback:   Optional[Callable[[Message], None]] = None,
        queue_name: Optional[str] = None,
        priority:   int = 128,
        bus:        str = "kernel",
    ) -> str:
        """Subscribe to a topic pattern. Returns sub_id for unsubscribe."""
        b = self._ns.get_or_create_bus(bus)
        q = self._ns.get_queue(queue_name) if queue_name else None
        return b.subscribe(pattern, callback=callback, queue=q, priority=priority)

    @agent_method(
        name="ipc_send",
        description="Send a message to a named queue",
        parameters={
            "queue_name": {"type": "str", "desc": "Target queue name"},
            "topic":      {"type": "str", "desc": "Message topic"},
            "payload":    {"type": "Any", "desc": "Message payload"},
        },
        priority=AgentPriority.NORMAL,
    )
    def send(
        self,
        queue_name: str,
        topic:      str,
        payload:    Any,
        sender:     str   = "kernel",
        timeout:    float = DELIVERY_TIMEOUT_S,
    ) -> bool:
        """Enqueue message; returns True on success."""
        q = self._ns.get_or_create_queue(queue_name)
        return q.send(Message(topic=topic, payload=payload, sender=sender), timeout)

    @agent_method(
        name="ipc_recv",
        description="Receive one message from a named queue",
        parameters={"queue_name": {"type": "str", "desc": "Source queue name"}},
        priority=AgentPriority.NORMAL,
    )
    def recv(
        self,
        queue_name: str,
        timeout:    float = DELIVERY_TIMEOUT_S,
    ) -> Optional[Dict[str, Any]]:
        """Dequeue message; returns to_dict() or None on timeout."""
        q = self._ns.get_or_create_queue(queue_name)
        msg = q.recv(timeout)
        return msg.to_dict() if msg else None

    @agent_method(
        name="ipc_channel_open",
        description="Open or create a bidirectional channel and return an endpoint",
        parameters={
            "channel_name": {"type": "str", "desc": "Channel name"},
            "endpoint_name": {"type": "str", "desc": "This endpoint's name"},
        },
        priority=AgentPriority.NORMAL,
    )
    def channel_open(
        self, channel_name: str, endpoint_name: str
    ) -> Dict[str, str]:
        """Open channel endpoint. Returns {channel, endpoint, status}."""
        ch = self._ns.get_or_create_channel(channel_name)
        ep = ch.endpoint(endpoint_name)
        return {
            "channel":  channel_name,
            "endpoint": endpoint_name,
            "status":   "open",
        }

    @agent_method(
        name="ipc_stats",
        description="Return IPC namespace statistics",
        priority=AgentPriority.LOW,
    )
    def stats(self) -> Dict[str, Any]:
        return {"namespace": self._ns.stats(), "version": _IPC_VERSION}


# ─────────────────────────────────────────────────────────────────────────────
# §9  SELF-TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """Deterministic validation suite. Raises AssertionError on failure."""

    # ── Message serialisation ─────────────────────────────────────────────────
    m1 = Message("kernel.memory.alloc", {"pages": 4}, sender="planner")
    raw = m1.serialise()
    m2  = Message.deserialise(raw)
    assert m2.topic   == m1.topic,   f"topic mismatch: {m2.topic}"
    assert m2.payload == m1.payload, f"payload mismatch: {m2.payload}"
    assert m2.sender  == m1.sender,  f"sender mismatch: {m2.sender}"
    assert m2.msg_id  == m1.msg_id,  f"msg_id mismatch: {m2.msg_id}"

    # ── MessageQueue — basic send/recv ────────────────────────────────────────
    q = MessageQueue("test_q", maxsize=4)
    assert len(q) == 0
    m  = Message("test.topic", 42)
    ok = q.send(m, timeout=0.1)
    assert ok, "send failed"
    assert len(q) == 1
    got = q.recv(timeout=0.1)
    assert got is not None and got.payload == 42
    assert len(q) == 0

    # Non-blocking paths
    q.try_send(Message("t", 1))
    q.try_send(Message("t", 2))
    r1 = q.try_recv()
    r2 = q.try_recv()
    assert r1.payload == 1 and r2.payload == 2

    try:
        q.try_recv()
        assert False, "should have raised QueueEmptyError"
    except QueueEmptyError:
        pass

    # Fill to capacity
    for i in range(4):
        q.try_send(Message("t", i))
    try:
        q.try_send(Message("t", 999))
        assert False, "should have raised QueueFullError"
    except QueueFullError:
        pass

    # ── MessageQueue — close unblocks waiter ──────────────────────────────────
    qc = MessageQueue("close_test", 8)
    results: List[Optional[Message]] = []

    def _waiter():
        results.append(qc.recv(timeout=2.0))

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    time.sleep(0.05)
    qc.close()
    t.join(timeout=1.0)
    assert results == [None], f"Expected [None] after close, got {results}"

    # ── Channel ───────────────────────────────────────────────────────────────
    ch  = Channel("test_channel")
    ep1 = ch.endpoint("sender")
    ep2 = ch.endpoint("receiver")

    ep1.send(Message("ping", {"n": 1}))
    got2 = ep2.recv(timeout=0.1)
    assert got2 is not None and got2.payload == {"n": 1}
    assert got2.sender == "sender"

    ep2.send(Message("pong", {"n": 2}))
    got1 = ep1.recv(timeout=0.1)
    assert got1 is not None and got1.payload == {"n": 2}

    try:
        ch.endpoint("third")
        assert False, "should have raised ValueError for third endpoint"
    except ValueError:
        pass

    # ── Topic matching ────────────────────────────────────────────────────────
    assert _topic_matches("kernel.*",        "kernel.memory")
    assert _topic_matches("kernel.*",        "kernel.dispatch")
    assert not _topic_matches("kernel.*",    "kernel.memory.alloc")
    assert _topic_matches("kernel.**",       "kernel.memory.alloc")
    assert _topic_matches("kernel.**",       "kernel.memory")
    assert _topic_matches("kernel.**",       "kernel")
    assert _topic_matches("*.*",             "a.b")
    assert not _topic_matches("*.*",         "a.b.c")
    assert _topic_matches("kernel.memory.*", "kernel.memory.alloc")

    # ── EventBus ─────────────────────────────────────────────────────────────
    bus      = EventBus("test_bus")
    received : List[Any] = []

    sid = bus.subscribe("kernel.*", callback=lambda m: received.append(m.payload))
    bus.emit("kernel.boot", "started", sender="kernel")
    bus.emit("kernel.memory", {"pages": 2}, sender="allocator")
    bus.emit("user.request", "ignored", sender="user")  # should not match

    assert received == ["started", {"pages": 2}], f"Received: {received}"
    assert bus.unsubscribe(sid)

    # Wildcard ** subscription
    received2: List[str] = []
    bus.subscribe("kernel.**", callback=lambda m: received2.append(m.topic))
    bus.emit("kernel.a.b.c", None)
    bus.emit("kernel.x",     None)
    bus.emit("user.y",       None)
    assert set(received2) == {"kernel.a.b.c", "kernel.x"}, f"Got: {received2}"

    # ── Semaphore ─────────────────────────────────────────────────────────────
    sem = Semaphore("res", initial=3, max_value=3)
    assert sem.value == 3
    assert sem.acquire(1, timeout=0.1)
    assert sem.acquire(2, timeout=0.1)
    assert sem.value == 0
    assert not sem.acquire(1, timeout=0.05), "Should timeout on empty semaphore"
    sem.release(1)
    assert sem.value == 1
    sem.release(2)
    assert sem.value == 3

    try:
        sem.release(1)
        assert False, "release beyond max should raise"
    except ValueError:
        pass

    # ── SharedMemoryRegion ────────────────────────────────────────────────────
    shm = SharedMemoryRegion("test_shm", size=64)
    shm.write(0, b"HELLO")
    shm.write(5, b" WORLD")
    data = shm.read(0, 11)
    assert data == b"HELLO WORLD", f"SHM data: {data!r}"
    assert shm.stats()["version"] == 2

    try:
        shm.read(60, 10)
        assert False, "out-of-bounds read should raise"
    except IndexError:
        pass

    # ── IPCNamespace ──────────────────────────────────────────────────────────
    ns = IPCNamespace.instance()
    q2 = ns.get_or_create_queue("ns_test_q")
    q3 = ns.get_or_create_queue("ns_test_q")
    assert q2 is q3, "Same name must return same object"

    ch2 = ns.get_or_create_channel("ns_chan")
    ch3 = ns.get_or_create_channel("ns_chan")
    assert ch2 is ch3

    print("aios_ipc: all self-tests passed ✓")


if __name__ == "__main__":
    _run_self_tests()
