"""Deep Agents Code - Interactive AI coding assistant."""
# [해설][모듈 헤더] 패키지 `deepagents_code`의 최상위 `__init__`. CLI 클라이언트
# [해설]   프로세스와 LangGraph 서버 프로세스 양쪽에서 import되므로 가볍게 유지된다.
# [해설] 하는 일: (1) 디버그 로그 버퍼 설치와 로깅 설정, (2) `__version__` 노출, (3)
# [해설]   `cli_main`을 모듈 `__getattr__`로 지연 제공.
# [해설] 콘솔 스크립트 `dcode`/`deepagents-code`는 `deepagents_code:cli_main`을 가리키며
# [해설]   호출 시점에 `main.py`의 `cli_main`으로 해석된다.
# [해설][설계] 지연 import는 `config`, `widgets` 같은 하위 모듈만 import할 때
# [해설]   `main.py`의 argparse·시그널 초기화 비용을 피하기 위함이다.
# [해설] 관련: `main.py`(진입점 본체), `__main__.py`(`python -m`), `_invocation.py`(실행
# [해설]   이름), `_debug.py`/`_debug_buffer.py`(로깅, 이 레인 밖).

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from deepagents_code._debug import configure_debug_logging
from deepagents_code._debug_buffer import install_log_buffer
from deepagents_code._home_error import DeepAgentsHomeError
from deepagents_code._version import __version__

if TYPE_CHECKING:
    from collections.abc import Callable

# [해설][설계] 순서가 중요하다. 항상 켜진 메모리 버퍼(디버그 콘솔용)를 먼저 붙여야
# [해설]   이어지는 `configure_debug_logging`의 경고가 버퍼에 잡힌다. 패키지 로거는 자식
# [해설]   모듈이 로그를 내기 전에 설정되어야 한다.
install_log_buffer(logging.getLogger(__name__))  # noqa: RUF067  # attach the always-on tail first so warnings from configure_debug_logging are captured
configure_debug_logging(logging.getLogger(__name__))  # noqa: RUF067  # package logger must be configured before child modules emit logs; sets the final level over the buffer's INFO floor

# [해설] 공개 이름은 `__version__`과 `cli_main`뿐이다. `cli_main`은 아래 `__getattr__`이
# [해설]   지연 해석하므로 `noqa: F822`가 붙어 있다.
__all__ = [
    "__version__",
    "cli_main",  # noqa: F822  # resolved lazily by __getattr__
]


# [해설][설계] PEP 562 모듈 `__getattr__`. `from deepagents_code import cli_main`이나
# [해설]   `deepagents_code.cli_main` 접근 시점에야 `deepagents_code.main`을 import한다.
# [해설][주의] `DeepAgentsHomeError`(DEEPAGENTS_HOME이 잘못되었거나 홈을 찾지 못함)가
# [해설]   나면 트레이스백 대신 메시지를 출력하고 종료 코드 2로 끝나는 대체 `cli_main`을
# [해설]   돌려준다. 정상 `cli_main`이 아니라는 점에 유의.
def __getattr__(name: str) -> Callable[[], None]:
    """Lazy import for `cli_main` to avoid loading `main.py` at package import.

    `main.py` pulls in `argparse`, signal handling, and other startup machinery
    that isn't needed when submodules like `config` or `widgets` are
    imported directly.

    Returns:
        The requested callable.

    Raises:
        AttributeError: If *name* is not a lazily-provided attribute.

    Note:
        Any import error other than an unresolvable profile location
        propagates unchanged; `DeepAgentsHomeError` is reported as a message
        plus exit 2 instead of a traceback.
    """
    if name == "cli_main":
        try:
            from deepagents_code.main import cli_main
        except DeepAgentsHomeError as exc:
            # `_paths` resolves the profile at import, so a bad DEEPAGENTS_HOME
            # or an unresolvable home surfaces here. Report it as a message
            # rather than a traceback: the user has a value to fix, and every
            # module that imports `_paths` would fail the same way.
            message = str(exc)

            def cli_main() -> None:
                print(f"dcode: {message}", file=sys.stderr)  # noqa: T201
                raise SystemExit(2)

        return cli_main
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
