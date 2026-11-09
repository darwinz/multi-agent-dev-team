"""Error hierarchy and transient/permanent classifiers for LLM, tool and Chroma calls."""

from __future__ import annotations

import errno
from typing import Literal

ErrorKind = Literal["transient", "permanent"]

try:  # httpx ships with the provider SDKs; don't hard-fail if it is absent
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


class AgentError(Exception):
    """Base for errors raised by the resilience layer; wraps the original exception."""

    def __init__(self, message: str = "", *, original: BaseException | None = None):
        super().__init__(message)
        self.original = original

    def __str__(self) -> str:
        if self.original is not None:
            return f"{type(self.original).__name__}: {self.original}"
        return super().__str__()


class TransientError(AgentError):
    """A retryable failure that outlasted its retries."""

    def __init__(self, message: str = "", *, original: BaseException | None = None, attempts: int = 0):
        super().__init__(message, original=original)
        self.attempts = attempts


class PermanentError(AgentError):
    """A failure that retrying cannot fix."""


class CircuitOpenError(TransientError):
    """A circuit breaker is open; the call was not attempted."""

    def __init__(self, message: str = "", *, breaker: str = "", original: BaseException | None = None,
                 attempts: int = 0):
        super().__init__(message, original=original, attempts=attempts)
        self.breaker = breaker


_TRANSIENT_LLM_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
    "OverloadedError", "ServiceUnavailableError",
}
_TRANSIENT_ERRNOS = {errno.EAGAIN, errno.EBUSY, errno.EINTR, errno.ETIMEDOUT, errno.EMFILE, errno.ENFILE}


def _status(exc: BaseException) -> int | None:
    s = getattr(exc, "status_code", None)
    return s if isinstance(s, int) else None


def _is_httpx_transport(exc: BaseException) -> bool:
    return httpx is not None and isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def classify_llm_error(exc: BaseException) -> ErrorKind:
    s = _status(exc)
    if (isinstance(exc, (TransientError, TimeoutError, ConnectionError))
            or (s is not None and (s in {408, 409, 429} or s >= 500))
            or type(exc).__name__ in _TRANSIENT_LLM_NAMES
            or _is_httpx_transport(exc)):
        return "transient"
    return "permanent"


def classify_tool_error(exc: BaseException) -> ErrorKind:
    if isinstance(exc, (TransientError, TimeoutError, ConnectionError)):
        return "transient"
    if isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS:
        return "transient"
    return "permanent"


_CHROMA_CONNECT_FAILURE = "Could not connect to a Chroma server"


def _is_chroma_connect_failure(exc: BaseException) -> bool:
    """chromadb's HttpClient reports a refused/failed connection as a bare ValueError raised while
    handling the httpx transport error (chromadb 1.5.9, api/client.py get_user_identity)."""
    if not isinstance(exc, ValueError):
        return False
    inner = exc.__cause__ or exc.__context__
    return (_CHROMA_CONNECT_FAILURE in str(exc)
            or (inner is not None and (_is_httpx_transport(inner) or isinstance(inner, ConnectionError))))


def classify_chroma_error(exc: BaseException) -> ErrorKind:
    s = _status(exc)
    if (isinstance(exc, (TransientError, TimeoutError, ConnectionError))
            or _is_httpx_transport(exc)
            or _is_chroma_connect_failure(exc)
            or (s is not None and (s >= 500 or s == 429))):
        return "transient"
    return "permanent"
