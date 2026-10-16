"""Same-origin relay to the decision_lab service, so its UI can be embedded.

`ciel.py` reaches the service server-to-server, and that path works no matter
where the browser is. The *labeling* UI is a different problem: it has to run in
the browser, and there the service is unreachable — it binds loopback only, the
host firewall drops its port, and above all this site is served over HTTPS while
the service speaks plain HTTP, so an https page is not allowed to frame it.
Opening the port would not fix that last one.

So the embedded page never crosses an origin. `static/decision_lab.html` frames
`/lab` on *this* server (`PREFIX`), and `server.py` relays each request here to
the service. The address comes from `ciel.base_url()` — the same reader the
prediction adapter uses — so the embedded UI and the adapter cannot end up
pointed at two different services once `DECISION_LAB_URL` changes.

Stdlib `urllib`, like `ciel.py`: this is a handful of requests, not a reason to
take on an HTTP client dependency.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import ciel

# Must match the routes `server.py` registers. Reading and relaying are the only
# things that know this path, so a change here is a change to both.
PREFIX = "/lab"

STREAM_CHUNK = 8192

# "Auto-label all" streams one progress event per OpenRouter call, so a short
# read timeout would cut a batch off mid-run. 300s matches the proxy_read_timeout
# the rest of this host's vhosts use: long enough for a batch, bounded so a hung
# upstream surfaces as an error rather than a request that never returns.
DEFAULT_TIMEOUT = 300.0

# Hop-by-hop and framing headers belong to the transport, not to the caller:
# urllib derives Host from the URL it dials and Content-Length from the body it
# sends. accept-encoding is dropped deliberately — bodies are relayed verbatim,
# so a gzip response would reach the browser without the header needed to decode
# it.
_DROPPED_HEADERS = frozenset({
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "accept-encoding",
    "keep-alive",
    "te",
    "upgrade",
})

# `tab` is Decision Lab's own deep-link parameter (its app.js honors it). Adding
# it here rather than upstream keeps this repo the only place that knows the UI
# is embedded — and the script checks before rewriting, so the reload it triggers
# cannot loop.
_REDIRECT_SNIPPET = """
<script data-decision-lab-deep-link>
(function () {
  var url = new URL(location.href);
  if (url.searchParams.get("tab")) return;
  url.searchParams.set("tab", "label");
  location.replace(url.pathname + "?" + url.searchParams.toString() + url.hash);
})();
</script>
"""

# The upstream UI is light-styled; without this a dark-mode browser paints its
# form controls dark against its own light background.
_STYLE_SNIPPET = "<style>:root{color-scheme:light}</style>"

_HEAD_END = b"</head>"


class RelayError(RuntimeError):
    """The service could not be reached at all — as opposed to answering an error.

    An upstream 4xx/5xx is *not* this: that is the service talking, and the page
    should show what it said. This is only raised when nothing answered.
    """


@dataclass(frozen=True)
class RelayResponse:
    """What came back from the service, ready to hand to the browser."""

    status: int
    body: Iterator[bytes]
    content_type: str | None = None


def forwardable_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """The caller's headers, minus the ones the transport owns.

    Case-insensitive, because HTTP header names are: the browser's
    `X-Decision-Lab-Log` must survive as the opt-in that turns a call into
    training data.
    """
    return {name: value for name, value in headers.items() if name.lower() not in _DROPPED_HEADERS}


def _open(request: urllib.request.Request, timeout: float) -> Any:
    """The one place this module touches the network — the seam tests substitute."""
    return urllib.request.urlopen(request, timeout=timeout)


def _content_type(response: Any) -> str | None:
    headers = getattr(response, "headers", None)
    return None if headers is None else headers.get("content-type")


def _close_quietly(response: Any) -> None:
    close = getattr(response, "close", None)
    if close is not None:
        close()


def _chunks(response: Any) -> Iterator[bytes]:
    """Relay the upstream body as it arrives instead of buffering it whole.

    Streamed on purpose: one progress event per labeled item is only useful if it
    reaches the browser while the batch is still running.
    """
    try:
        while True:
            chunk = response.read(STREAM_CHUNK)
            if not chunk:
                return
            yield chunk
    finally:
        _close_quietly(response)


def _relabel(body: bytes, add_redirect: bool) -> bytes:
    """Add the Label deep-link and the light color scheme to the served page.

    A body without the marker tag is returned untouched: an unmodified page that
    works beats a rewritten one whose scripts never ran.
    """
    if _HEAD_END not in body:
        return body
    snippet = (_REDIRECT_SNIPPET if add_redirect else "") + _STYLE_SNIPPET
    return body.replace(_HEAD_END, snippet.encode("utf-8") + _HEAD_END, 1)


def relay(
    method: str,
    path: str,
    query: str = "",
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    rewrite_html: bool = False,
) -> RelayResponse:
    """Send one request to the service and return its answer.

    ``path`` is the *upstream* path (`/api/status`), not this relay's own; the
    caller strips `PREFIX` before calling.

    Raises:
        RelayError: nothing answered at `ciel.base_url()`.
    """
    url = f"{ciel.base_url()}{path}" + (f"?{query}" if query else "")
    request = urllib.request.Request(
        url,
        data=body,
        headers=dict(headers or {}),
        method=method.upper(),
    )
    try:
        response = _open(request, DEFAULT_TIMEOUT)
    except urllib.error.HTTPError as exc:
        # The service answering its own 4xx/5xx is information, not a relay
        # failure: "no pending items to label" is the UI's business. Pass the
        # status and body through so the page shows the service's real message.
        try:
            return RelayResponse(exc.code, iter([exc.read()]), _content_type(exc))
        finally:
            _close_quietly(exc)
    except urllib.error.URLError as exc:
        raise RelayError(
            f"could not reach Decision Lab at {ciel.base_url()}: {exc.reason}"
        ) from exc

    content_type = _content_type(response)
    if rewrite_html and content_type is not None and content_type.startswith("text/html"):
        # Only worth emitting the redirect when the page did not ask for a view
        # already; the styling applies either way.
        add_redirect = "tab" not in urllib.parse.parse_qs(query)
        try:
            return RelayResponse(
                response.status,
                iter([_relabel(response.read(), add_redirect)]),
                content_type,
            )
        finally:
            _close_quietly(response)
    return RelayResponse(response.status, _chunks(response), content_type)
