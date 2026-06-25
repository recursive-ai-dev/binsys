"""binsys — create and run filesystem images like VMs."""

from __future__ import annotations

# ── Custom Exceptions ──────────────────────────────────────────────────────────


class BinSysError(Exception):
    """Base exception for all binsys errors."""
    pass


class ConfigError(BinSysError):
    """Configuration-related errors."""
    pass


class DependencyError(BinSysError):
    """Missing system dependencies."""
    pass


class SecurityError(BinSysError):
    """Security-related errors (encryption, authentication)."""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed."""
    pass


class RateLimitError(SecurityError):
    """Rate limiting triggered."""
    pass


class ImageError(BinSysError):
    """Image-related errors."""
    pass


class MountError(BinSysError):
    """Mount/umount errors."""
    pass


class DownloadError(BinSysError):
    """Download-related errors."""
    pass


class ValidationError(BinSysError):
    """Input validation errors."""
    pass
