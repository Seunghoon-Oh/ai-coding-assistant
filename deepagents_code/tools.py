"""Custom tools for the agent."""

# [해설] 모듈 역할: dcode 가 에이전트에게 직접 주는 커스텀 도구 3종 - fetch_url(URL
# [해설]   본문을 마크다운으로), web_search(Tavily 웹 검색), get_current_thread_id(현재
# [해설]   스레드 ID).
# [해설] 실행 위치: 도구 함수는 LangGraph 서버 프로세스에서 실행된다(server_graph.py 의
# [해설]   _build_tools 가 목록을 만들어 create_cli_agent 에 넘김). --acp 경로는 main.py
# [해설]   가 같은 도구를 직접 조립한다.
# [해설] 주요 심볼: fetch_url, web_search, create_web_search_tool(작업공간 키에 묶인
# [해설]   변형), is_web_search_tool, get_current_thread_id.
# [해설] 이 모듈의 보안 중심은 SSRF 방어다. URL 은 LLM 이 정하므로 프롬프트 인젝션으로
# [해설]   내부 주소를 찌를 수 있다. _validate_url(검증) + _pinned_dns(검증한 IP로 연결
# [해설]   고정) + _fetch_with_redirects(리다이렉트마다 재검증) 3단 방어를 갖는다.
# [해설] 파일 읽기/쓰기/셸 같은 기본 도구는 여기 없다. 그것들은 SDK 의
# [해설]   FilesystemMiddleware/백엔드가 제공한다(agent.py 참고).
# [해설] 관련 레인 모듈: agent.py(도구 목록 소비, 승인 규칙), subagents.py, offload*.py.

from __future__ import annotations

import contextlib
import functools
import ipaddress
import logging
import socket
import threading
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import urljoin, urlparse

from langchain_core.tools import tool
from langgraph.config import get_config
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from langchain_core.tools import BaseTool
    from tavily import TavilyClient

logger = logging.getLogger(__name__)

# [해설] 'Tavily 클라이언트를 아직 만들지 않음'을 나타내는 센티널. None('키 없음')과
# [해설]   구분하려고 별도 객체를 쓴다.
_UNSET = object()
# [해설] 지연 생성되는 프로세스 전역 Tavily 클라이언트 캐시. _get_tavily_client 가
# [해설]   채운다.
_tavily_client: TavilyClient | object | None = _UNSET

_WEB_SEARCH_MARKER = "deepagents_web_search"
"""Tool-metadata key marking a workspace-bound `web_search` variant.

Read by `is_web_search_tool`, the same way MCP read-only hints are read off
tool metadata, so a variant does not have to be registered anywhere.
"""

_WEB_SEARCH_TOKEN = object()
"""Value `is_web_search_tool` requires under `_WEB_SEARCH_MARKER`.

A module-private object rather than `True` so the marker cannot be forged: MCP
tool metadata is deserialized JSON, which can carry the key but never this
identity. Callers therefore need no separate "is this tool remote?" guard.
"""

# [해설] fetch_url 이 허용하는 스킴. file://, ftp:// 등은 _validate_url 에서 거부된다.
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_MAX_FETCH_REDIRECTS = 5

# Maintainer note: `deepagents-talon` imports `web_search` and `fetch_url`
# directly from this module. Keep their names, signatures, and return/error dict
# shapes stable unless `deepagents-talon` is migrated in the same change.

# Module-level lock guarding the urllib3 connection-factory monkeypatch used by
# `_pinned_dns`. The patch is process-global, so serializing fetches keeps
# concurrent calls from clobbering each other's pinned IP set.
# [해설][주의] _pinned_dns 는 urllib3 의 전역 함수를 임시로 바꿔치기(monkeypatch)한다.
# [해설]   그래서 락으로 직렬화하며, 동시에 여러 fetch_url 이 돌아도 사실상 한 번에
# [해설]   하나씩 네트워크 요청이 나간다(성능 한계).
_dns_pin_lock = threading.Lock()


# [해설] SSRF 방어에 의한 의도적 거부와, 다른 이유의 ValueError 를 구분하기 위한 전용
# [해설]   예외. fetch_url 이 category="validation" 으로 변환한다.
class _UrlValidationError(ValueError):
    """Raised by `_validate_url` for scheme/DNS/SSRF-blocked URLs.

    Distinguishes intentional SSRF-guard rejections from incidental
    `ValueError`s raised elsewhere in the fetch path (e.g., markdown
    conversion).
    """


# [해설] 공인 IP 가 아닌 모든 주소(사설망, 루프백, 링크로컬=클라우드 메타데이터
# [해설]   169.254.169.254 등)를 막는 판정 함수.
# [해설][설계] IPv4-mapped IPv6 와 6to4 는 먼저 안쪽 IPv4 로 풀어서 검사한다 - IPv6 로
# [해설]   감싸 사설 주소를 숨기는 우회를 막기 위함.
def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if `ip` belongs to a non-publicly-routable range.

    Rejects: private (RFC1918/ULA), loopback, link-local (including cloud
    IMDS at `169.254.169.254`), reserved, multicast, unspecified
    (`0.0.0.0`/`::`), and anything `ipaddress` does not consider globally
    routable (catches benchmarking, documentation, and similar ranges the
    explicit predicates miss).

    IPv4-mapped IPv6 (`::ffff:a.b.c.d`) and 6to4 (`2002::/16`) are unwrapped
    to their underlying IPv4 address before the checks so that private
    space tunneled inside an IPv6 wrapper is still caught — e.g.,
    `::ffff:127.0.0.1` and `2002:a9fe:a9fe::1` (6to4 over IMDS) both
    evaluate as blocked.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


# [해설] SSRF 1단계: 스킴/호스트명을 확인하고 DNS 를 풀어, 나온 모든 IP 가 공인 주소일
# [해설]   때만 IP 목록을 돌려준다. 하나라도 막힌 주소면 거부.
# [해설][주의] 여기서 DNS 를 한 번 풀기 때문에 이후 연결 시 다시 풀면 DNS rebinding(검사
# [해설]   시점과 사용 시점의 불일치, TOCTOU)이 가능하다. 그 틈을 _pinned_dns 가 메운다.
def _validate_url(url: str) -> list[str]:
    """Reject URLs that target private/internal/metadata addresses.

    Resolves the URL's hostname and rejects any URL whose hostname resolves
    to a private, loopback, link-local (includes cloud IMDS at
    `169.254.169.254`), reserved, multicast, or unspecified IP — including
    such addresses wrapped in IPv4-mapped IPv6 (`::ffff:...`) or 6to4
    (`2002::/16`). This is the SSRF guard required because the URL is
    supplied by an LLM agent and may originate from prompt-injected content.

    Note:
        This function resolves DNS once. The HTTP client must be pinned to
        the returned IP list (see `_pinned_dns`) to close the TOCTOU window
        against attacker-controlled DNS (rebinding).

    Args:
        url: Candidate URL to validate.

    Returns:
        The list of validated IP strings the hostname resolves to.

            Callers should pin the outgoing connection to one of these IPs.

    Raises:
        _UrlValidationError: If the URL is malformed, uses a disallowed
            scheme, fails DNS resolution, or resolves to a blocked address.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        msg = f"URL scheme not allowed: {parsed.scheme!r} (must be http or https)"
        raise _UrlValidationError(msg)

    hostname = parsed.hostname
    if not hostname:
        msg = "URL is missing a hostname"
        raise _UrlValidationError(msg)

    try:
        encoded_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        msg = f"Could not encode hostname {hostname!r} as IDNA: {exc}"
        raise _UrlValidationError(msg) from exc

    try:
        infos = socket.getaddrinfo(
            encoded_hostname,
            None,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        msg = f"Could not resolve hostname {hostname!r}: {exc}"
        raise _UrlValidationError(msg) from exc

    validated_ips: list[str] = []
    for info in infos:
        # `sockaddr[0]` may include an IPv6 scope id (`fe80::1%eth0`); strip
        # it before parsing so `ipaddress.ip_address` never raises.
        raw_ip = str(info[4][0]).split("%", 1)[0]
        ip = ipaddress.ip_address(raw_ip)
        if _is_blocked_ip(ip):
            logger.warning(
                "SSRF guard blocked URL %r: hostname %r resolves to %s",
                url,
                hostname,
                ip,
            )
            msg = (
                f"URL hostname {hostname!r} resolves to blocked address {ip} "
                "(private, loopback, link-local, reserved, or non-global range)"
            )
            raise _UrlValidationError(msg)
        validated_ips.append(raw_ip)

    if not validated_ips:
        msg = f"Hostname {hostname!r} resolved to no addresses"
        raise _UrlValidationError(msg)

    return validated_ips


# [해설] SSRF 2단계 컨텍스트 매니저 _pinned_dns 의 데코레이터. with 블록 동안만 연결
# [해설]   패치가 유지된다.
@contextlib.contextmanager
# [해설] with 블록 안에서 hostname 으로 나가는 연결을 검증된 IP 로만 보내도록 urllib3 의
# [해설]   create_connection 을 임시 교체한다. requests 가 호스트명을 다시 풀어 다른 IP
# [해설]   로 가는 것을 막는다.
# [해설][주의] 프로세스 전역 패치이므로 _dns_pin_lock 으로 보호한다. finally 에서 반드시
# [해설]   원복한다.
def _pinned_dns(hostname: str, allowed_ips: list[str]) -> Iterator[None]:
    """Force outgoing urllib3 connections for `hostname` to use `allowed_ips`.

    Patches `urllib3.util.connection.create_connection` for the duration of
    the context so that `requests` cannot re-resolve `hostname` to a
    different IP than the one `_validate_url` vetted (defends against DNS
    rebinding TOCTOU). The patch is process-global, so the module lock
    serializes concurrent fetches.

    Args:
        hostname: The exact hostname (already IDNA-encoded by the caller)
            whose resolution must be pinned.
        allowed_ips: The IPs `_validate_url` confirmed are safe to connect
            to. Tried in order; the first that accepts the connection wins.
    """
    from urllib3.util import connection as urllib3_connection

    with _dns_pin_lock:
        original = urllib3_connection.create_connection

        def patched(
            address: tuple[str, int], *args: Any, **kwargs: Any
        ) -> socket.socket:
            host, port = address[0], address[1]
            if host != hostname:
                return original(address, *args, **kwargs)
            last_exc: OSError | None = None
            for ip in allowed_ips:
                try:
                    return original((ip, port), *args, **kwargs)
                except OSError as exc:
                    last_exc = exc
            assert last_exc is not None  # noqa: S101  # loop body guarantees this
            raise last_exc

        urllib3_connection.create_connection = patched  # ty: ignore[invalid-assignment]  # signature matches at runtime
        try:
            yield
        finally:
            urllib3_connection.create_connection = original


# [해설] markdownify 가 재귀 한도에 걸릴 때 쓰는 대체 텍스트 추출기. script/style 등은
# [해설]   건너뛰어, 신뢰할 수 없는 페이지의 JS/CSS 가 본문으로 새지 않게 한다.
class _TextExtractor(HTMLParser):
    """Extract text content from HTML as a markdownify fallback.

    The character data inside raw-text elements (`script`, `style`,
    `noscript`, `template`) is skipped so the fallback never emits
    JavaScript or CSS source from the fetched (untrusted) page as page
    content.
    """

    # Tags whose character data is never page content. Suppressed via an
    # explicit allowlist of skipped tags rather than trying to detect script
    # payloads after the fact.
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],  # noqa: ARG002  # required by HTMLParser override
    ) -> None:
        """Enter a raw-text element so its data is skipped."""
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """Leave a raw-text element."""
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        """Collect non-empty, whitespace-collapsed text outside skipped tags."""
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        """Return extracted text fragments separated by blank lines."""
        return "\n\n".join(self.parts)


# [해설] HTML -> 마크다운. 깊게 중첩된 HTML 에서 RecursionError 가 나면 _TextExtractor
# [해설]   로 대체하고, 그것마저 실패하면 빈 문자열을 돌려준다(크래시 방지).
def _html_to_markdown_content(html: str, markdownify: Callable[[str], str]) -> str:
    """Convert HTML to markdown, falling back to plain text on recursion.

    Args:
        html: Raw HTML to convert.
        markdownify: The `markdownify.markdownify` callable, injected so this
            module avoids an eager top-level import of the optional dependency.

    Returns:
        Markdown content, or text extracted from the HTML if markdown
        conversion exceeds the recursion limit. Returns an empty string if
        the text-extraction fallback itself fails.
    """
    try:
        return markdownify(html)
    except RecursionError:
        logger.warning(
            "markdownify hit recursion depth; falling back to text extraction",
            exc_info=True,
        )

    # Best-effort plain-text extraction. Guard it so a failure here (e.g. the
    # same pathological input that exhausted markdownify's recursion) cannot
    # re-introduce the uncaught crash this fallback exists to prevent.
    try:
        parser = _TextExtractor()
        parser.feed(html)
        parser.close()
    except Exception:  # fallback is best-effort; must never propagate
        logger.warning("text-extraction fallback failed", exc_info=True)
        return ""
    return parser.get_text()


# [해설] 키가 없을 때 예외 대신 모델이 읽을 수 있는 에러 dict 를 돌려준다. 모델이 상황을
# [해설]   이해하고 다음 행동을 고르게 하려는 도구 설계 방식이다.
def _missing_tavily_key_error(query: object) -> dict[str, object]:
    """Return the payload the model sees when no Tavily key is configured.

    Shared by the built-in and workspace-bound variants: `is_web_search_tool`
    treats them as one tool, so they have to fail identically.

    Returns:
        Error payload naming the env var to set.
    """
    return {
        "error": "Tavily API key not configured. "
        "Please set TAVILY_API_KEY environment variable.",
        "query": query,
    }


def _missing_package_error(exc: ImportError) -> dict[str, str]:
    """Return the payload the model sees when an optional package is absent.

    Returns:
        Error payload naming the missing package.
    """
    return {"error": f"Required package not installed: {exc.name}."}


# [해설] Tavily 클라이언트 지연 싱글턴. 키는 config.credentials 에서 읽는다(import
# [해설]   시점이 아니라 첫 호출 시점에 평가).
def _get_tavily_client() -> TavilyClient | None:
    """Get or initialize the lazy Tavily client singleton.

    Returns:
        TavilyClient instance, or None if API key is not configured.
    """
    global _tavily_client  # noqa: PLW0603  # Module-level cache requires global statement
    if _tavily_client is not _UNSET:
        return _tavily_client  # ty: ignore[invalid-return-type]  # narrowed by sentinel check

    from deepagents_code.config import credentials

    if credentials.has_tavily:
        from tavily import TavilyClient as _TavilyClient

        _tavily_client = _TavilyClient(api_key=credentials.tavily_api_key)
    else:
        _tavily_client = None
    return _tavily_client


# [해설] 작업공간(workspace)별 Tavily 키에 묶인 web_search 변형을 만든다. 서버가 여러
# [해설]   작업공간을 다룰 때 키가 섞이지 않게 하려는 용도로 보인다(추정; 호출부는
# [해설]   server_graph.py 의 _build_tools).
# [해설][설계] functools.wraps(web_search) 로 스키마를 기본 web_search 와 똑같이
# [해설]   유지하고, 메타데이터 마커로 is_web_search_tool 이 둘을 같은 도구로 인식하게
# [해설]   한다.
def create_web_search_tool(api_key: str) -> BaseTool:
    """Bind web search to one workspace credential.

    The schema is taken from `web_search` via `functools.wraps` so the built-in
    and workspace-bound variants can never present different arguments. The two
    also have to fail the same way: `is_web_search_tool` treats them as one, so
    a missing package or an unusable key must return the payload the model can
    act on rather than raising.

    Returns:
        Workspace-bound web search tool.
    """
    # Built on first use and reused: a per-call client would open a fresh
    # connection pool and repeat the TLS handshake for every search.
    client: TavilyClient | None = None

    @tool("web_search")
    @functools.wraps(web_search)
    def workspace_web_search(**kwargs: Any) -> object:
        nonlocal client
        if not api_key:
            return _missing_tavily_key_error(kwargs.get("query"))
        if client is None:
            try:
                from tavily import TavilyClient as _TavilyClient

                client = _TavilyClient(api_key=api_key)
            except ImportError as exc:
                return _missing_package_error(exc)
        return _search_with_tavily(client, **kwargs)

    workspace_web_search.metadata = {
        **(workspace_web_search.metadata or {}),
        _WEB_SEARCH_MARKER: _WEB_SEARCH_TOKEN,
    }
    return workspace_web_search


# [해설] 어떤 도구가 web_search(기본 또는 작업공간 변형)인지 판정한다. 마커 값을 모듈
# [해설]   전용 객체(_WEB_SEARCH_TOKEN)로 두었기 때문에 MCP 가 JSON 메타데이터로 위조할
# [해설]   수 없다.
def is_web_search_tool(candidate: object) -> bool:
    """Return whether `candidate` is a built-in or workspace-bound search tool.

    Returns:
        `True` for the module-level tool or any variant the factory marked.
    """
    if candidate is web_search:
        return True
    metadata = getattr(candidate, "metadata", None) or {}
    return metadata.get(_WEB_SEARCH_MARKER) is _WEB_SEARCH_TOKEN


# [해설] LangChain @tool 데코레이터: 함수를 모델이 호출 가능한 도구로 바꾼다. docstring
# [해설]   이 도구 설명으로 쓰이므로 docstring 도 모델에게 보이는 프롬프트의 일부다.
@tool
# [해설] LangGraph 실행 설정(get_config)에서 현재 thread_id 를 꺼내 돌려준다.
# [해설]   LangSmith/MCP 도구가 같은 대화를 가리키게 할 때 쓴다.
def get_current_thread_id() -> str:
    """Get the current Deep Agents thread ID for LangSmith or MCP tooling.

    Returns:
        The current `configurable.thread_id`, or an explanatory message if missing.
    """
    thread_id = get_config().get("configurable", {}).get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return "No current thread ID is available."


# [해설] 기본 web_search 도구. 이 함수에는 @tool 이 붙어 있지 않고, main.py(ACP 경로
# [해설]   _run_acp_cli_async)만 일반 함수 그대로 도구 목록에 넣는다(server_graph.py 는
# [해설]   create_web_search_tool 의 변형을 쓴다). 함수 -> 도구 변환은 langgraph
# [해설]   ToolNode 가 create_tool 로 한다(tool_node.py:789). 변형은 @tool("web_search")
# [해설]   로 직접 감싼다.
# [해설] Annotated + Field 의 description 이 모델에게 보이는 인자 설명이 된다.
def web_search(  # noqa: ANN201  # Return type depends on dynamic tool configuration
    query: Annotated[
        str,
        Field(description="The search query (be specific and detailed)."),
    ],
    max_results: Annotated[
        int,
        Field(description="Number of results to return."),
    ] = 5,
    topic: Annotated[
        Literal["general", "news", "finance"],
        Field(
            description=(
                'Search topic type: "general" for most queries, "news" for '
                'current events, or "finance".'
            )
        ),
    ] = "general",
    include_raw_content: Annotated[
        bool,
        Field(
            description=(
                "Include full page content (uses more tokens). Prefer `fetch_url` "
                "for a single URL."
            )
        ),
    ] = False,
):
    """Search the web for current information.

    Returns:
        Search hits with title, URL, snippet, and score.
    """
    client = _get_tavily_client()
    if client is None:
        return _missing_tavily_key_error(query)
    return _search_with_tavily(
        client,
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


# [해설] 기본/작업공간 변형이 공유하는 실제 검색 실행부. Tavily 와 requests 의 예외를
# [해설]   에러 dict 로 번역해 모델이 읽게 한다.
def _search_with_tavily(
    client: TavilyClient,
    *,
    query: str,
    max_results: int,
    topic: Literal["general", "news", "finance"],
    include_raw_content: bool,
) -> object:
    """Execute a Tavily search with the standard error translation.

    Returns:
        Search hits or a translated error payload.
    """
    try:
        import requests
        from tavily import (
            BadRequestError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            UsageLimitExceededError,
        )
        from tavily.errors import ForbiddenError, TimeoutError as TavilyTimeoutError
    except ImportError as exc:
        return _missing_package_error(exc)

    try:
        return client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
        # Tavily-specific exceptions
        BadRequestError,
        ForbiddenError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        UsageLimitExceededError,
    ) as e:
        return {"error": f"Web search error: {e!s}", "query": query}


# [해설] 단일 URL 을 가져와 마크다운으로 변환하는 도구. 실패는 예외 대신 {error, url,
# [해설]   category} dict 로 돌려준다(category: validation | redirects | network).
# [해설][흐름] 1) 선택 패키지 import 2) _fetch_with_redirects 로 안전하게 가져오기 3)
# [해설]   _html_to_markdown_content 로 변환 4) 메타데이터와 함께 반환.
def fetch_url(
    url: Annotated[
        str,
        Field(description="The URL to fetch (must be a valid HTTP/HTTPS URL)."),
    ],
    timeout: Annotated[
        int,
        Field(description="Request timeout in seconds."),
    ] = 30,
) -> dict[str, Any]:
    """Fetch a URL and return the page content as markdown.

    Returns:
        Fetched page markdown plus status metadata.
    """
    try:
        import requests
        from markdownify import markdownify
    except ImportError as exc:
        return _missing_package_error(exc)

    try:
        response = _fetch_with_redirects(url, timeout=timeout)
    except _UrlValidationError as e:
        return {
            "error": f"Fetch URL error: {e!s}",
            "url": url,
            "category": "validation",
        }
    except requests.exceptions.TooManyRedirects as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url, "category": "redirects"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url, "category": "network"}

    markdown_content = _html_to_markdown_content(response.text, markdownify)
    if not markdown_content:
        logger.warning(
            "fetch_url produced empty content for %s (status %s)",
            response.url,
            response.status_code,
        )
    return {
        "url": str(response.url),
        "markdown_content": markdown_content,
        "status_code": response.status_code,
        "content_length": len(markdown_content),
    }


# [해설] SSRF 3단계: 리다이렉트를 requests 에 맡기지 않고(allow_redirects=False) 직접
# [해설]   따라가며 매 홉마다 _validate_url + _pinned_dns 를 다시 적용한다.
# [해설][주의] session.trust_env = False 로 환경 프록시를 끈다. 프록시가 대상 호스트를
# [해설]   따로 풀면 DNS 고정이 무력화되기 때문이다. 대신 프록시 뒤에 있는 환경에서는
# [해설]   fetch_url 이 외부로 못 나갈 수 있다.
def _fetch_with_redirects(url: str, *, timeout: int) -> Any:  # noqa: ANN401  # requests.Response, but kept dynamic to avoid eager import
    """Fetch `url`, re-validating each redirect hop against the SSRF guard.

    Each hop is validated by `_validate_url` and its connection pinned to
    the validated IP via `_pinned_dns`. Caps at `_MAX_FETCH_REDIRECTS`
    redirects (so up to `_MAX_FETCH_REDIRECTS + 1` total hops counting the
    initial request). Network/HTTP errors propagate as
    `requests.exceptions.RequestException` (or its subclasses).

    Args:
        url: Initial URL to fetch.
        timeout: Per-request timeout in seconds.

    Returns:
        The final `requests.Response` for the non-redirect terminal hop.

    Raises:
        _UrlValidationError: If any hop fails SSRF validation or returns a
            3xx without a `Location` header.
        requests.exceptions.TooManyRedirects: If the redirect cap is exceeded.
    """
    import requests

    current_url = url
    session = requests.Session()
    # DNS pinning only protects the direct target connection. Environment
    # proxies resolve the target separately, so they must be disabled here.
    session.trust_env = False
    for _hop in range(_MAX_FETCH_REDIRECTS + 1):
        validated_ips = _validate_url(current_url)
        hostname = urlparse(current_url).hostname
        # `_validate_url` raises if hostname is missing, so this is non-None.
        assert hostname is not None  # noqa: S101  # invariant from _validate_url
        encoded_hostname = hostname.encode("idna").decode("ascii")

        with _pinned_dns(encoded_hostname, validated_ips):
            response = session.get(
                current_url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
                allow_redirects=False,
            )

        # 300-399 covers every redirect class. `requests.Response.is_redirect`
        # also checks for a `Location` header, which would hide malformed 3xx
        # responses — so we check the raw status code instead.
        if 300 <= response.status_code < 400:  # noqa: PLR2004  # HTTP redirect class
            location = response.headers.get("Location")
            if not location:
                msg = (
                    f"Redirect response (status {response.status_code}) at "
                    f"{current_url!r} is missing a Location header"
                )
                raise _UrlValidationError(msg)
            current_url = urljoin(current_url, location)
            continue

        response.raise_for_status()
        return response

    msg = f"Exceeded {_MAX_FETCH_REDIRECTS} redirects starting from {url!r}"
    raise requests.exceptions.TooManyRedirects(msg)
