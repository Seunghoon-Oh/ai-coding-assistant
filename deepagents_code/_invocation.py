"""Resolution of the command name this process was launched with.

Hints that tell the user how to resume a thread have to echo a command the user
can actually paste back. `dcode` is only one of the names that reach this code:
the package ships both `deepagents-code` and `dcode` console scripts, and
per-project shims (a renamed symlink in `~/.local/bin` pointing at a worktree's
`bin/dcode`) are a common way to run several checkouts side by side. Hardcoding
`dcode` tells those users to run a command that may not exist.

`sys.argv[0]` holds the answer, because the kernel passes the pathname given to
`execve` to the interpreter rather than the symlink target, so a shim invoked as
`abc` reports `abc`. It is not always meaningful, though, so `invoked_name`
falls back to `DEFAULT_INVOKED_NAME` whenever the value is missing or does not
look like a command a user could have typed.
"""

# [해설][모듈 헤더] 이 프로세스가 어떤 이름(`dcode`, `deepagents-code`, 사용자 shim)으로
# [해설]   실행됐는지 알아내는 작은 유틸리티. CLI 클라이언트 프로세스에서 쓴다.
# [해설] 목적은 '종료 시 재개 힌트'가 사용자가 실제로 붙여넣을 수 있는 명령이 되게 하는
# [해설]   것이다(`main.py`의 `_render_teardown_thread_hints`가 `invoked_name()`을
# [해설]   사용).
# [해설] 주요 심볼: `invoked_name`(이름 결정), `log_nonstandard_invoked_name`(비표준
# [해설]   이름 기록), `DEFAULT_INVOKED_NAME`, `STANDARD_INVOKED_NAMES`.
# [해설][설계] 시작 경로에 있으므로 import를 가볍게 유지한다. `main.py`의 `cli_main`은
# [해설]   이 모듈을 함수 안에서 지연 import한다.
# [해설] 관련: `_env_vars.py`(`INVOKED_AS`, `DEBUG`), `_debug_buffer.py`(디버그 콘솔
# [해설]   버퍼), `__main__.py`.
from __future__ import annotations

import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import PurePath

from deepagents_code._env_vars import DEBUG, INVOKED_AS, is_env_truthy

logger = logging.getLogger(__name__)

# [해설] 이름을 알 수 없을 때의 폴백이며 가장 흔한 콘솔 스크립트 이름이다.
DEFAULT_INVOKED_NAME = "dcode"
"""Command name assumed when the launch name cannot be determined."""

# [해설][주의] `pyproject.toml`의 `[project.scripts]`와 손으로 맞춰 둔 중복 목록이다.
# [해설]   스크립트를 추가하면 여기도 고쳐야 하며, 어긋남은 `test_invocation.py`의 drift
# [해설]   가드가 잡는다.
STANDARD_INVOKED_NAMES = frozenset({"dcode", "deepagents-code"})
"""Console scripts shipped in `pyproject.toml` (`[project.scripts]`).

Anything else — a per-checkout shim or a user alias — is non-standard and gets a
one-line note in the Debug Console at launch (see `log_nonstandard_invoked_name`).
Duplicated here by hand because this module must stay import-light; reading the
installed entry points at runtime would be slower and can disagree with the shim
the user actually typed. `test_invocation.py` has a drift guard against
`pyproject.toml`.
"""

# [해설] 비정상적으로 긴 값(주입 의심 포함)을 걸러내는 상한.
_MAX_NAME_LENGTH = 64

# [해설][설계] `argv[0]`과 환경 변수는 외부에서 오는 값이고 결과가 복사 가능한 명령에
# [해설]   들어가므로, 이스케이프 대신 허용 목록(allowlist) 정규식으로 모양을 제한한다.
_SAFE_NAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
"""Plausible console-script names: no separators, spaces, or shell metacharacters.

`sys.argv[0]` and the environment are supplied by whatever started the process,
and the resolved name is rendered into a copy-pasteable command, so the shape is
allowlisted rather than escaped.
"""

_WINDOWS_EXECUTABLE_SUFFIX = ".exe"


# [해설] 후보 이름을 명령 이름으로 정리한다. Windows `.exe`는 떼고, 빈 값·너무 긴
# [해설]   값·`.py` 파일·`python*` 인터프리터 이름·허용 정규식 밖의 값은 `None`으로
# [해설]   거부한다.
def _sanitize(raw: str) -> str | None:
    """Return `raw` as a command name, or `None` when it is not plausible.

    Args:
        raw: A candidate name (an `argv[0]` basename or an env-var value).

    Returns:
        The cleaned command name, or `None` when the value cannot be a console
            script the user typed — empty, absurdly long, a Python source file
            (`python -m deepagents_code` reports `__main__.py`), an interpreter
            name, or anything outside `_SAFE_NAME_RE`.
    """
    name = raw.strip()
    if name.lower().endswith(_WINDOWS_EXECUTABLE_SUFFIX):
        # Windows console scripts are `.exe` wrappers; the user types the stem.
        name = name[: -len(_WINDOWS_EXECUTABLE_SUFFIX)]
    if not name or len(name) > _MAX_NAME_LENGTH:
        return None
    if name.endswith(".py") or name.lower().startswith("python"):
        return None
    if not _SAFE_NAME_RE.match(name):
        return None
    return name


# [해설] `invoked_name`: 명령 이름 결정. 우선순위는 1) 환경 변수 `INVOKED_AS`(자동
# [해설]   업데이트 재시작 때 `main.py`의 `_restart_current_process`가 이름을 넘겨
# [해설]   주려고 설정), 2) `sys.argv[0]`의 basename, 3) `DEFAULT_INVOKED_NAME`.
# [해설][주의] 프로세스 수명 동안 한 번만 계산한다(`lru_cache`). 테스트에서 `sys.argv`나
# [해설]   환경을 바꾸면 `invoked_name.cache_clear()`가 필요하다.
@lru_cache(maxsize=1)
def invoked_name() -> str:
    """Return the command name this process was launched with.

    Cached: `sys.argv[0]` and the launch environment are fixed for the life of
    the process. Tests that vary either must call `invoked_name.cache_clear()`.

    Returns:
        The console-script name the user invoked (for example `dcode`,
            `deepagents-code`, or a shim name), or `DEFAULT_INVOKED_NAME` when it
            cannot be determined.
    """
    override = os.environ.get(INVOKED_AS)
    if override is not None:
        name = _sanitize(override)
        if name is not None:
            return name
        logger.debug("Ignoring implausible %s value", INVOKED_AS)
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        name = _sanitize(PurePath(argv0).name)
        if name is not None:
            return name
    return DEFAULT_INVOKED_NAME


# [해설] `log_nonstandard_invoked_name`: 표준 이름이 아니면 디버그 콘솔에만 한 줄
# [해설]   남긴다. 레벨은 디버그 모드 여부로 고른다(버퍼가 패키지 로거를 INFO 바닥으로
# [해설]   두므로 디버그가 꺼져 있으면 INFO로 남겨야 콘솔에 보인다). `main.py`의
# [해설]   `cli_main` 초반에서 호출된다.
# [해설][주의] 한 번만 실행되므로 테스트는
# [해설]   `log_nonstandard_invoked_name.cache_clear()`가 필요하다.
@lru_cache(maxsize=1)
def log_nonstandard_invoked_name() -> None:
    r"""Note a non-standard launch name in the Debug Console, once per process.

    Cached so repeated calls cannot repeat the note; tests that vary the launch
    environment must call `log_nonstandard_invoked_name.cache_clear()`. Shim
    users launch through a name this package does not ship (see
    `STANDARD_INVOKED_NAMES`), and a wrong resume hint is otherwise impossible to
    trace back to how the name was resolved.

    The level is chosen so the note is never user-facing but always reaches the
    in-app Debug Console (`Ctrl+\\`): the in-memory buffer floors the package
    logger at `INFO` and its handler passes `DEBUG` (`_debug_buffer`), so when
    `DEEPAGENTS_CODE_DEBUG` is off a `DEBUG` record would be filtered before the
    buffer saw it — `INFO` still prints nothing to the terminal because the
    buffer is the only handler in the chain. When debug mode is on, `DEBUG` is
    used so the note also lands in the debug log file.
    """
    name = invoked_name()
    if name in STANDARD_INVOKED_NAMES:
        return
    logger.log(
        logging.DEBUG if is_env_truthy(DEBUG) else logging.INFO,
        "Invoked as non-standard command %r; resume hints will use this name",
        name,
    )
