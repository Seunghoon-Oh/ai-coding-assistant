"""Agent middleware covering what MCP tools need around every call.

The tools themselves come out of `langchain.mcp` unmodified. Everything this
app wants on top — an argument quirk, and translating a dead-token failure into
something a model and a user can act on — sits here instead, so the adapter
stays the plain conversion it is and this behavior applies to any MCP tool the
agent is given.
"""

# [해설] 이 모듈은 에이전트가 MCP 도구를 호출할 때마다 끼어드는 LangChain 미들웨어 한
# [해설]   개(mcp_tool_middleware)를 제공한다.
# [해설] 목적은 두 가지다. 1) 모델이 선택 인자에 빈 문자열을 채워 보낸 것을 '생략'으로
# [해설]   정규화(normalize_mcp_arguments)
# [해설]   2) 호출 시간 제한과 '토큰 만료(재인증 필요)' 실패를 모델·사용자가 이해할 수
# [해설]   있는 ToolMessage로 변환.
# [해설] 어떤 도구가 MCP 도구인지는 mcp_tools.py의 _build_mcp_tool이 metadata에 붙인
# [해설]   표식(_deepagents_code_mcp)으로 안다.
# [해설]   그래서 이 미들웨어는 MCP가 아닌 도구 호출은 건드리지 않고 그대로 통과시킨다.
# [해설] 장착 위치: agent.py의 create_cli_agent가 MCP 도구가 있을 때 미들웨어 목록에
# [해설]   추가한다(타임아웃은 resolve_mcp_tool_timeout 설정).
# [해설]   langchain의 wrap_tool_call 데코레이터로 만들어 도구 호출 앞뒤를 감싼다.
# [해설] 주의: 인자 정규화는 도구 객체 쪽(mcp_tools.py의 normalized_call)에도 같은
# [해설]   함수가 걸려 있다. 미들웨어 없이 도구만 직접 호출해도 같은 규칙이
# [해설]   적용되며(test_mcp_tools.py의 test_call_normalizes_optional_empty_string), 두
# [해설]   번 정규화해도 결과는 같다.
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from deepagents_code.config_manifest import MCP_TOOL_TIMEOUT_SECONDS_DEFAULT

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest

logger = logging.getLogger(__name__)

# [해설] mcp_tools.py가 도구 metadata에 기록하는 키와 같은 문자열이어야 한다(두 파일이
# [해설]   문자열 상수를 따로 가진다).
_MCP_MARKER = "_deepagents_code_mcp"
_MCP_SERVER = "_deepagents_code_mcp_server"


def _is_mcp_tool(request: ToolCallRequest) -> bool:
    """Return whether this call is going to a tool adapted from MCP."""
    metadata = getattr(request.tool, "metadata", None) or {}
    return bool(metadata.get(_MCP_MARKER))


# [해설] 선택(optional) 문자열 인자에 ''가 오면 키 자체를 뺀다. 필수 인자와 명시적
# [해설]   비문자열 타입(정수·불리언·배열)은 그대로 둔다.
# [해설]   타입이 없거나(oneOf/$ref) properties에 없는 키도 모호하다고 보고 뺀다. None은
# [해설]   서버가 판단하도록 건드리지 않는다.
def normalize_mcp_arguments(
    arguments: dict[str, Any],
    input_schema: Any,  # noqa: ANN401  # raw JSON Schema dict from the MCP tool
) -> dict[str, Any]:
    """Drop empty-string values for optional MCP tool params.

    Some MCP servers (e.g. Slack's `slack_search_public_and_private`) validate
    optional ID-typed params with `value is not a channel ID` when the model
    fills them in with `""` instead of omitting them. JSON-Schema-derived
    models happily accept `""` for an optional string, so the request reaches
    the server and comes back as a generic failure.

    Treat `""` for a non-required string field as "omitted", so the server sees
    the payload it would have for a field the model genuinely skipped. Required
    fields pass through unchanged, so the server's own missing-field error still
    runs. Only `""` is normalized; `None` is left to the server.

    Args:
        arguments: Arguments the model produced for this call.
        input_schema: The MCP tool's input schema (raw JSON Schema dict).

    Returns:
        A new argument dict, or `arguments` unchanged when nothing was dropped.
    """
    if not isinstance(input_schema, dict):
        return arguments
    required = set(input_schema.get("required") or ())
    properties = input_schema.get("properties") or {}
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if value != "" or key in required:  # noqa: PLC1901  # telling "" from 0/False/[] is the point
            cleaned[key] = value
            continue
        prop = properties.get(key)
        prop_type = prop.get("type") if isinstance(prop, dict) else None
        is_string_typed = prop_type == "string" or (
            isinstance(prop_type, list) and "string" in prop_type
        )
        # Three drop conditions converge here:
        #   - explicit string type (the original Slack-style failure mode);
        #   - missing `type` (oneOf/anyOf/$ref or untyped — treat as ambiguous
        #     and drop, since a server rejects `""` for any ID-shaped slot);
        #   - key absent from `properties` entirely (model invented a field).
        # Anything with an explicit non-string `type` is kept: `""` cannot be a
        # valid integer/bool/array, so it was the model's mistake to send, and
        # the server's own validation says so more clearly than we could.
        if isinstance(prop, dict) and not is_string_typed and prop_type is not None:
            cleaned[key] = value
    if cleaned.keys() != arguments.keys():
        logger.debug(
            "MCP arg normalize: dropped empty-string keys %s",
            sorted(set(arguments) - set(cleaned)),
        )
    return cleaned


# [해설] 미들웨어 팩토리. 반환된 _wrap이 모든 도구 호출을 받아 MCP 도구에만 인자
# [해설]   정규화와 시간 제한을 적용한다.
# [해설][흐름] 1) MCP 도구가 아니면 그대로 통과 2) 인자를 정규화해 request를 교체 3)
# [해설]   asyncio.wait_for로 시간 제한을 걸고 handler 실행
# [해설]   4) ToolException은 이미 서버 메시지가 담긴 실패이므로 그대로 다시 던진다 5)
# [해설]   시간 초과는 상태 'error'인 ToolMessage로 변환
# [해설]   6) 그 밖의 예외는 mcp_auth.find_reauth_required로 재인증 필요 여부를 확인해,
# [해설]   맞으면 로그인 안내 ToolMessage로, 아니면 원래 예외를 다시 던진다.
def mcp_tool_middleware(
    timeout_seconds: float = MCP_TOOL_TIMEOUT_SECONDS_DEFAULT,
) -> AgentMiddleware:
    """Build the middleware wrapping every MCP tool call.

    Returns:
        Middleware that normalizes arguments and reports a stale-token failure
            as a tool message naming the login command.
    """

    # [해설][주의] 시간 초과 메시지가 '서버 쪽에서 작업이 계속 실행 중일 수 있다'고
    # [해설]   알리는 이유: wait_for는 클라이언트 쪽 대기만 취소하므로 재시도가 중복
    # [해설]   작업이 될 수 있다.
    @wrap_tool_call(name="MCPToolMiddleware")
    async def _wrap(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        tool = request.tool
        if tool is None or not _is_mcp_tool(request):
            return await handler(request)

        server = (tool.metadata or {}).get(_MCP_SERVER, "?")
        arguments = normalize_mcp_arguments(
            request.tool_call.get("args") or {},
            getattr(request.tool, "args_schema", None),
        )
        request = request.override(
            tool_call={**request.tool_call, "args": arguments},
        )

        try:
            return await asyncio.wait_for(
                handler(request),
                timeout=timeout_seconds,
            )
        # A `ToolException` already reached its tool-local handler, which turned
        # the server's own error content into a failed `ToolMessage`. Re-raising
        # would bury an actionable instruction ("use the X tool instead") under
        # a generic wrapper.
        except ToolException:
            raise
        except TimeoutError:
            tool_name = request.tool_call.get("name") or getattr(tool, "name", "?")
            logger.warning(
                "MCP tool call timed out: server=%r tool=%r timeout=%s seconds",
                server,
                tool_name,
                timeout_seconds,
            )
            return ToolMessage(
                content=(
                    f"MCP tool call to server {server!r}, tool {tool_name!r}, "
                    f"timed out after {timeout_seconds:g} seconds. The operation "
                    "may still be running server-side; retrying may duplicate "
                    "work."
                ),
                tool_call_id=request.tool_call["id"],
                name=tool_name,
                status="error",
            )
        except Exception as exc:
            from deepagents_code.mcp_auth import find_reauth_required

            reauth = find_reauth_required(exc)
            if reauth is None:
                raise
            # Tokens existed but no longer refresh. The model cannot fix that,
            # so end the call with the instruction the *user* can act on rather
            # than an opaque transport error.
            logger.warning("MCP server %r needs re-authentication: %s", server, reauth)
            return ToolMessage(
                content=str(reauth),
                tool_call_id=request.tool_call["id"],
                name=request.tool_call.get("name"),
                status="error",
            )

    return _wrap


__all__ = ["mcp_tool_middleware", "normalize_mcp_arguments"]
