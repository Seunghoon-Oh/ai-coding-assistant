"""Durable, server-authoritative thread workspace bindings."""

# [해설] 역할: "스레드(thread_id) <-> 작업디렉터리(cwd) + 접근 정책"의 영속 바인딩.
# [해설]   서버가 권위(authoritative)를 가지며,
# [해설]   클라이언트가 보내는 cwd/정책 주장은 신뢰하지 않고 여기 저장된 값과 대조한다.
# [해설] 저장 위치: 체크포인트와 같은 SQLite 파일(sessions.db, _database_path). 별도
# [해설]   테이블 2개:
# [해설]   dcode_thread_workspaces(바인딩 본체), dcode_workspace_snapshots(거부 시 보여
# [해설]   줄 비교용 스냅샷).
# [해설]   대화 체크포인트 테이블과는 분리되어 있어, 바인딩 마이그레이션이 대화 기록을
# [해설]   건드리지 않는다(_bind 주석 참고).
# [해설] 프로세스: DB 읽기/쓰기(bind/get/require)는 서버 프로세스에서 일어난다.
# [해설]   canonical_fingerprint는 지문 계산용이라
# [해설]   클라이언트 쪽 코드(_server_config.py의 지문 메서드)에서도 import된다.
# [해설] 진입 심볼: bind_thread_workspace(새 스레드 바인딩 생성/검증, offload_api.py가
# [해설]   import), get_thread_workspace(조회),
# [해설]   require_thread_workspace(실행 요청 검증, server_graph.py의 make_graph가
# [해설]   호출), resolve_workspace(신원+지문 계산).
# [해설] 관련 모듈: _server_config.py(정책 필드 분류와 지문),
# [해설]   workspace_diagnostics.py(거부 사유 스냅샷), server_graph.py.
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, TypedDict, cast

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code.workspace_diagnostics import (
    WorkspaceDiagnostics,
    WorkspaceSnapshot,
    diff_snapshots,
    snapshot_for_payload,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# [해설][설계] 바인딩 스키마 버전(현재 4). v4에서 지문이 "접근 정책 지문"과 "런타임
# [해설]   지문"으로 나뉘었다.
# [해설]   아래 문자열이 버전별 차이와 마이그레이션 규칙을 설명한다(옛 지문이 정확히
# [해설]   일치할 때만 제자리 이전).
_SCHEMA_VERSION = 4
_STRICT_LEGACY_SCHEMA_VERSION = 3
"""Binding schema generation.

Version 4 splits the single config fingerprint into a durable policy
fingerprint (trust/tool/sandbox/approval + workspace identity) and a full
runtime fingerprint (which also covers model settings). Version 3 resolves
project policy per workspace. Version 2 used the launch config's fingerprint
for every directory. `_bind` migrates older rows in place only when the old
full fingerprint still matches exactly — proof nothing policy-relevant
changed; otherwise the row is rejected as unprovable. Missing fingerprint
information is never treated as permission equivalence.
"""
_MAX_PATH_LENGTH = 4096
_MAX_CONFIG_LENGTH = 64_000

logger = logging.getLogger(__name__)


# [해설] 클라이언트와 LangGraph 런타임 컨텍스트가 주고받는 "공개용" 작업공간 서술자.
# [해설]   지문(fingerprint)은 서버 안에만
# [해설] 두기 위해 to_payload에서 제외된다.
class WorkspacePayload(TypedDict):
    """JSON-safe workspace descriptor carried in LangGraph runtime context."""

    schema_version: int
    workspace_id: str
    cwd: str
    project_root: str | None
    generation: int
    resource_key: str
    config_fingerprint: str


# [해설] DB 한 행에 대응하는 불변 바인딩 객체. workspace_id는 (cwd, project_root)의
# [해설]   해시, resource_key는 신원+정책의 해시다.
@dataclass(frozen=True)
class WorkspaceBinding:
    """Server-authoritative workspace and resource policy for one thread."""

    schema_version: int
    workspace_id: str
    cwd: str
    project_root: str | None
    generation: int
    resource_key: str
    config_fingerprint: str
    workspace_config_json: str
    policy_fingerprint: str = ""
    """Durable access-policy fingerprint; empty on pre-v4 rows until migrated."""

    runtime_fingerprint: str = ""
    """Full runtime identity fingerprint; empty on pre-v4 rows until migrated."""

    def to_payload(self) -> WorkspacePayload:
        """Return the public runtime-context representation.

        Fingerprints stay server-side: the payload is workspace *identity*
        (which the client echoes verbatim), while policy and runtime
        compatibility are checked against the server-resolved config, not a
        client-claimed fingerprint.
        """
        payload = asdict(self)
        payload.pop("workspace_config_json")
        payload.pop("policy_fingerprint")
        payload.pop("runtime_fingerprint")
        return cast("WorkspacePayload", payload)

    def workspace_config(self) -> dict[str, Any]:
        """Return the persisted, server-authoritative resource policy."""
        return cast("dict[str, Any]", json.loads(self.workspace_config_json))


# [해설] 작업공간 주장이 서버 정책과 충돌할 때의 예외. diagnostics에는 허용된 필드만
# [해설]   담긴 구조화 설명이 붙어
# [해설] 클라이언트 UI가 거부 이유를 보여 줄 수 있다.
class WorkspaceConflictError(RuntimeError):
    """A workspace claim or runtime conflicts with server resource policy."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: WorkspaceDiagnostics | None = None,
    ) -> None:
        """Initialize with the refusal message and optional safe diagnostics.

        Args:
            message: The refusal text; unchanged whether or not diagnostics
                are attached, so existing handlers stay compatible.
            diagnostics: Structured, allowlisted detail about the conflict.
        """
        super().__init__(message)
        self.diagnostics = diagnostics

    @classmethod
    def from_reason(
        cls,
        reason: str,
        *,
        diagnostics: WorkspaceDiagnostics | None = None,
    ) -> WorkspaceConflictError:
        """Build a workspace-hosting refusal with a stated reason.

        Args:
            reason: The cause, phrased to follow "because".
            diagnostics: Structured, allowlisted detail about the conflict.

        Returns:
            A conflict with the standard workspace-hosting message.
        """
        msg = f"Cannot host this workspace because {reason}."
        return cls(msg, diagnostics=diagnostics)


# [해설][흐름] DB 경로 결정: 환경변수 DEEPAGENTS_CODE_SERVER_DB_PATH(클라이언트가
# [해설]   server_manager.py의 _write_checkpointer에서 설정)가
# [해설]   있으면 그것, 없으면 sessions.get_db_path()(= 프로필 state 디렉터리의
# [해설]   sessions.db). 체크포인터와 같은 파일이다.
def _database_path() -> Path:
    value = os.environ.get(f"{SERVER_ENV_PREFIX}DB_PATH")
    if value:
        return Path(value)
    from deepagents_code.sessions import get_db_path

    return get_db_path()


# [해설][설계] 클라이언트가 보낸 경로를 신뢰하지 않고 검증한다: 비어 있지 않은 절대경로,
# [해설]   `..` 금지, 실제 존재하는 디렉터리,
# [해설] SDK의 validate_path 통과(Windows 제외). 심볼릭 링크는 resolve로 정규화한 뒤
# [해설]   다시 검사한다.
def _canonical_directory(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > _MAX_PATH_LENGTH:
        msg = f"workspace.{field} must be a non-empty absolute path"
        raise ValueError(msg)
    candidate = Path(value)
    if not candidate.is_absolute() or ".." in PurePath(value).parts:
        msg = f"workspace.{field} must be an absolute path without traversal"
        raise ValueError(msg)
    if os.name != "nt":
        from deepagents.backends.utils import validate_path

        validate_path(value)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        msg = f"workspace.{field} is unavailable: {value}"
        raise ValueError(msg) from exc
    if not resolved.is_dir():
        msg = f"workspace.{field} is not a directory: {value}"
        raise ValueError(msg)
    if os.name != "nt":
        from deepagents.backends.utils import validate_path

        validate_path(str(resolved))
    return resolved


# [해설] 정책 딕셔너리를 정렬된 JSON(+SHA-256)으로 만든다. 크기
# [해설]   상한(_MAX_CONFIG_LENGTH)을 넘으면 거부.
def canonical_workspace_config(value: object | None) -> tuple[str, str]:
    """Return bounded canonical JSON and its SHA-256 fingerprint.

    Raises:
        TypeError: If the configuration is not an object.
        ValueError: If it cannot be serialized or exceeds the size limit.
    """
    if value is None:
        value = {}
    if not isinstance(value, dict):
        msg = "workspace_config must be an object"
        raise TypeError(msg)
    try:
        serialized = _canonical_json(value)
    except (TypeError, ValueError) as exc:
        msg = "workspace configuration must be JSON serializable"
        raise ValueError(msg) from exc
    if len(serialized) > _MAX_CONFIG_LENGTH:
        msg = "workspace configuration is too large"
        raise ValueError(msg)
    return serialized, hashlib.sha256(serialized.encode()).hexdigest()


def _canonical_json(value: object) -> str:
    """Return JSON with consistent key ordering and spacing for fingerprinting."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# [해설] 정렬된 JSON 직렬화의 SHA-256. 키 순서에 영향받지 않는 지문이라
# [해설]   클라이언트/서버가 독립적으로 계산해도 같은 값이 나온다.
def canonical_fingerprint(value: object) -> str:
    """Fingerprint `value` with the canonical workspace serialization.

    Returns:
        The SHA-256 hex digest of the canonical JSON encoding.
    """
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


# [해설][흐름] 1) cwd 검증(_canonical_directory) 2) find_project_root로 프로젝트 루트
# [해설]   탐색 3) workspace_id = (cwd, project_root) 해시
# [해설][흐름] 4) policy_fingerprint = 정책 + 신원 해시(모델이 바뀌어도 불변) 5)
# [해설]   resource_key = workspace_id + policy_fingerprint
# [해설][흐름] 6) runtime_fingerprint는 호출자가 넘긴
# [해설]   config_fingerprint(ServerConfig.runtime_fingerprint)를 사용.
# [해설] 이 함수는 DB를 건드리지 않는다. 결과를 저장하는 것은 _bind다.
def resolve_workspace(
    cwd: object,
    workspace_config: object | None = None,
    *,
    config_fingerprint: str | None = None,
) -> WorkspaceBinding:
    """Resolve a client-supplied cwd into a canonical workspace binding.

    `cwd` is untrusted and is validated here. `workspace_config` is not: every
    caller passes server-resolved policy. A client claim is verified against
    server policy in `offload_api.workspace` and never reaches this function.

    Returns:
        A canonical, fingerprinted binding including the resource policy.
    """
    config_json, payload_fingerprint = canonical_workspace_config(workspace_config)
    config_fingerprint = config_fingerprint or payload_fingerprint
    canonical_cwd = _canonical_directory(cwd, field="cwd")
    from deepagents_code.project_utils import find_project_root

    project_root = find_project_root(canonical_cwd)
    if project_root is not None:
        project_root = _canonical_directory(str(project_root), field="project_root")
    workspace_id = canonical_fingerprint(
        {
            "cwd": str(canonical_cwd),
            "project_root": str(project_root) if project_root else None,
        }
    )
    # Durable access-policy compatibility: the persisted payload (policy)
    # fingerprinted together with the resolved workspace identity. Cosmetic
    # model settings and runtime-only fields are excluded, so a model change
    # does not invalidate the binding.
    policy_fingerprint = canonical_fingerprint(
        {
            "cwd": str(canonical_cwd),
            "policy": json.loads(config_json),
            "project_root": str(project_root) if project_root else None,
        }
    )
    # `resource_key` is the stable policy+identity key: it must NOT change when
    # only the runtime identity (model) does, or the payload comparison and
    # runtime cache would treat a permitted model switch as a new workspace.
    resource_key = canonical_fingerprint(
        {"workspace_id": workspace_id, "policy_fingerprint": policy_fingerprint}
    )
    return WorkspaceBinding(
        schema_version=_SCHEMA_VERSION,
        workspace_id=workspace_id,
        cwd=str(canonical_cwd),
        project_root=str(project_root) if project_root else None,
        generation=1,
        resource_key=resource_key,
        config_fingerprint=config_fingerprint,
        workspace_config_json=config_json,
        policy_fingerprint=policy_fingerprint,
        # The runtime fingerprint is the full (to_env) fingerprint the caller
        # passes as `config_fingerprint`; the payload-only fallback covers
        # callers that supply no server config.
        runtime_fingerprint=config_fingerprint,
    )


# [해설] 스키마를 멱등하게 만든다(CREATE IF NOT EXISTS + 빠진 컬럼 ALTER ADD). 호출마다
# [해설]   실행되므로 옛 DB도 자동으로 올라온다.
def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dcode_thread_workspaces (
            thread_id TEXT PRIMARY KEY NOT NULL,
            schema_version INTEGER NOT NULL,
            workspace_id TEXT NOT NULL,
            cwd TEXT NOT NULL,
            project_root TEXT,
            generation INTEGER NOT NULL,
            resource_key TEXT NOT NULL,
            config_fingerprint TEXT NOT NULL,
            workspace_config_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(dcode_thread_workspaces)").fetchall()
    }
    if "config_fingerprint" not in columns:
        conn.execute(
            "ALTER TABLE dcode_thread_workspaces "
            "ADD COLUMN config_fingerprint TEXT NOT NULL DEFAULT ''"
        )
    if "workspace_config_json" not in columns:
        conn.execute(
            "ALTER TABLE dcode_thread_workspaces "
            "ADD COLUMN workspace_config_json TEXT NOT NULL DEFAULT '{}'"
        )
    if "policy_fingerprint" not in columns:
        conn.execute(
            "ALTER TABLE dcode_thread_workspaces "
            "ADD COLUMN policy_fingerprint TEXT NOT NULL DEFAULT ''"
        )
    if "runtime_fingerprint" not in columns:
        conn.execute(
            "ALTER TABLE dcode_thread_workspaces "
            "ADD COLUMN runtime_fingerprint TEXT NOT NULL DEFAULT ''"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dcode_workspace_snapshots (
            thread_id TEXT PRIMARY KEY NOT NULL,
            snapshot_version INTEGER NOT NULL,
            snapshot_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


# [해설] sqlite3.Row -> WorkspaceBinding 변환.
def _row_binding(row: sqlite3.Row) -> WorkspaceBinding:
    return WorkspaceBinding(
        schema_version=row["schema_version"],
        workspace_id=row["workspace_id"],
        cwd=row["cwd"],
        project_root=row["project_root"],
        generation=row["generation"],
        resource_key=row["resource_key"],
        config_fingerprint=row["config_fingerprint"],
        workspace_config_json=row["workspace_config_json"],
        policy_fingerprint=row["policy_fingerprint"],
        runtime_fingerprint=row["runtime_fingerprint"],
    )


# [해설][설계] 스냅샷은 INSERT OR IGNORE다. 새 스레드 행을 만들 때만 기록하고, 거부된
# [해설]   재바인딩이 증거를 덮어쓰지 못하게 한다.
def _write_snapshot(
    conn: sqlite3.Connection, thread_id: str, snapshot: WorkspaceSnapshot
) -> None:
    """Persist a binding's comparison snapshot, or refuse to overwrite one.

    A rejected binding must never replace the recorded snapshot — it is the
    evidence for the next refusal — so this is `INSERT OR IGNORE` by design:
    only a *new* thread row introduces a snapshot. Snapshots carry only
    allowlisted policy values (see `workspace_diagnostics`).
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO dcode_workspace_snapshots (
            thread_id, snapshot_version, snapshot_json
        ) VALUES (?, ?, ?)
        """,
        (thread_id, snapshot.snapshot_version, snapshot.to_json()),
    )


# [해설] 스레드의 스냅샷 읽기. 스냅샷 도입 전의 옛 바인딩에는 없어서 None이다.
def _read_snapshot(
    conn: sqlite3.Connection, thread_id: str
) -> WorkspaceSnapshot | None:
    """Read a thread's persisted snapshot; a legacy row has none.

    Returns:
        The snapshot, or `None` when the binding predates snapshotting.
    """
    row = conn.execute(
        "SELECT snapshot_json FROM dcode_workspace_snapshots WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    return WorkspaceSnapshot.from_json(row[0])


# [해설] thread_id 없이 (workspace_id, config_fingerprint)로 스냅샷을 찾는다. 런타임
# [해설]   검증 단계는 바인딩만 들고 있기 때문이다.
def _snapshot_for_binding(
    conn: sqlite3.Connection, binding: WorkspaceBinding
) -> WorkspaceSnapshot | None:
    """Read the snapshot recorded for a binding's thread row.

    Runtime validation holds only the binding, not the thread id, so the row
    is located by workspace identity and policy fingerprint; a legacy row
    (which predates snapshots) or an unresolvable match yields `None`, which
    callers report as `snapshot_status="unavailable"`.

    Returns:
        The snapshot, or `None` when none was recorded for the binding.
    """
    row = conn.execute(
        """
        SELECT s.snapshot_json
        FROM dcode_workspace_snapshots s
        JOIN dcode_thread_workspaces w ON w.thread_id = s.thread_id
        WHERE w.workspace_id = ? AND w.config_fingerprint = ?
        """,
        (binding.workspace_id, binding.config_fingerprint),
    ).fetchone()
    if row is None:
        return None
    return WorkspaceSnapshot.from_json(row[0])


# [해설] 비동기 래퍼. SQLite 작업은 asyncio.to_thread로 넘겨 서버 이벤트 루프를 막지
# [해설]   않는다. server_graph.py의
# [해설] _resolve_bound_workspace_config가 드리프트 사유를 만들 때 호출한다.
async def get_snapshot_for_binding(
    binding: WorkspaceBinding,
) -> WorkspaceSnapshot | None:
    """Read the persisted comparison snapshot recorded for a binding.

    Returns:
        The snapshot, or `None` when the binding predates snapshotting.
    """

    def _snapshot() -> WorkspaceSnapshot | None:
        with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            _initialize(conn)
            return _snapshot_for_binding(conn, binding)

    return await asyncio.to_thread(_snapshot)


# [해설] "현재 스키마보다 옛 행인가"(지문이 없거나 schema_version이 낮음). 참이면
# [해설]   _bind가 제자리에서 이전을 시도한다.
def _is_migratable(existing: WorkspaceBinding) -> bool:
    """Whether a row's recorded policy predates the current schema.

    Older rows may contain launch-project policy instead of workspace policy.
    `_binding_differs` checks workspace identity and recorded session policy
    before allowing migration.

    Returns:
        `True` when the row has no fingerprint yet, or an older schema.
    """
    return not existing.config_fingerprint or existing.schema_version < _SCHEMA_VERSION


# [해설] 저장된 정책과 제안된 정책에서 값이 다른 필드 이름을 정렬해 돌려준다(진단용).
def _policy_drift_names(
    existing: WorkspaceBinding, proposed: WorkspaceBinding
) -> list[str]:
    """Name the durable policy fields that changed between two bindings.

    Compares the persisted policy payloads, tolerating `None`-vs-empty and
    excluding nothing — the payload already omits cosmetic model settings and
    runtime-only fields, so any difference here is real policy drift.

    Returns:
        The drifted policy field names, sorted; empty when policy matches.
    """
    bound_policy = existing.workspace_config()
    proposed_policy = proposed.workspace_config()
    return sorted(
        key
        for key in bound_policy.keys() | proposed_policy.keys()
        if bound_policy.get(key) != proposed_policy.get(key)
    )


# [해설][설계] 제안 바인딩이 저장된 것과 호환되지 않는가? 신원이 다르면 무조건 거부.
# [해설]   현재 행은 접근 정책 지문만 비교(모델 변경은 허용).
# [해설] 옛 행: v3 이상은 전체 지문이 정확히 같을 때만 허용, v2는 세션 정책 필드만 비교.
# [해설]   증명할 수 없으면 거부(fail closed).
def _binding_differs(existing: WorkspaceBinding, proposed: WorkspaceBinding) -> bool:
    """Whether the proposed binding is incompatible with the persisted one.

    Returns:
        `True` when the binding must be refused (see `_binding_conflict`).
    """
    if existing.workspace_id != proposed.workspace_id:
        return True
    if not _is_migratable(existing):
        # Current rows: only durable policy (and identity) invalidate a binding.
        # A full-runtime (model) change must not; the runtime fingerprint change
        # rebuilds the runtime rather than rebinding the thread.
        if existing.policy_fingerprint:
            return existing.policy_fingerprint != proposed.policy_fingerprint
        return existing.config_fingerprint != proposed.config_fingerprint
    if not existing.config_fingerprint:
        # Pre-fingerprint rows have no recorded policy to preserve.
        return False
    # A v3 row's old fingerprint covered the full runtime and policy. Exact
    # equality is the only proof that project policy did not change; unlike v2,
    # v3 recorded enough information to fail closed rather than infer safety
    # from the narrower session-policy payload.
    if existing.schema_version >= _STRICT_LEGACY_SCHEMA_VERSION:
        return existing.config_fingerprint != proposed.config_fingerprint
    if existing.config_fingerprint == proposed.config_fingerprint:
        return False
    from deepagents_code._server_config import SESSION_WORKSPACE_FIELDS

    bound_policy = existing.workspace_config()
    proposed_policy = proposed.workspace_config()
    return any(
        bound_policy.get(key) != proposed_policy.get(key)
        for key in SESSION_WORKSPACE_FIELDS
    )


# [해설] 거부 사유 문구 상수. server_graph.py의 _resolve_bound_workspace_config와 아래
# [해설]   _binding_conflict가 공유한다.
PROJECT_POLICY_DRIFT_REASON = (
    "the project's resolved policy differs from the policy recorded "
    "when this workspace was bound"
)
SERVER_CONFIG_DRIFT_REASON = (
    "the server configuration changed after this workspace was bound"
)


# [해설] 프로젝트 범위 정책(PROJECT_WORKSPACE_FIELDS) 중 바인딩 시점과 달라진 필드 이름.
# [해설]   값은 경로/불리언뿐이라 노출해도 안전하다.
def drifted_project_fields(
    bound_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
) -> list[str]:
    """Name the project-scoped fields that drifted from their binding.

    The refusal these feed is safe either way, but it is not diagnosable
    without the field names: the resolution reads the extension trust store on
    every call, so a transient read failure reports as a policy change. These
    values are paths and booleans, never secrets, so naming them is safe.

    Returns:
        The drifted field names, sorted; empty when the policy is unchanged.
    """
    from deepagents_code._server_config import PROJECT_WORKSPACE_FIELDS

    return sorted(
        key
        for key in PROJECT_WORKSPACE_FIELDS
        if bound_config.get(key) != current_config.get(key)
    )


# [해설] 거부 예외를 "만들어" 돌려준다(raise는 호출자가). 신원 불일치 / 옛 스키마 /
# [해설]   프로젝트 정책 드리프트 / 서버 설정 드리프트를 구분한다.
def _binding_conflict(
    thread_id: str,
    existing: WorkspaceBinding,
    proposed: WorkspaceBinding,
    snapshot: WorkspaceSnapshot | None,
) -> WorkspaceConflictError:
    """Build the binding refusal, attaching allowlisted drift diagnostics.

    Args:
        thread_id: The thread whose binding was refused.
        existing: The persisted binding.
        proposed: The rejected proposed binding; never persisted.
        snapshot: The persisted comparison snapshot, or `None` for a legacy
            binding that predates snapshotting.

    Returns:
        The conflict to raise; never raised here so the caller can log first.
    """
    if existing.workspace_id != proposed.workspace_id:
        return WorkspaceConflictError(
            f"thread {thread_id} is already bound to a different workspace",
            diagnostics=WorkspaceDiagnostics(
                category="bound_elsewhere",
                reason="bound to a different workspace",
            ),
        )
    proposed_snapshot = snapshot_for_payload(proposed.workspace_config())
    snapshot_status = "current" if snapshot is not None else "unavailable"
    if _is_migratable(existing):
        return WorkspaceConflictError.from_reason(
            SERVER_CONFIG_DRIFT_REASON,
            diagnostics=WorkspaceDiagnostics(
                category="config_drift",
                reason=SERVER_CONFIG_DRIFT_REASON,
                changes=diff_snapshots(snapshot, proposed_snapshot),
                snapshot_status=snapshot_status,
                binding_schema_version=existing.schema_version,
                server_schema_version=proposed.schema_version,
            ),
        )
    drifted = _policy_drift_names(existing, proposed)
    project_drift = drifted_project_fields(
        existing.workspace_config(), proposed.workspace_config()
    )
    changes = diff_snapshots(snapshot, proposed_snapshot, changed_names=drifted)
    if project_drift:
        return WorkspaceConflictError.from_reason(
            PROJECT_POLICY_DRIFT_REASON,
            diagnostics=WorkspaceDiagnostics(
                category="policy_drift",
                reason=PROJECT_POLICY_DRIFT_REASON,
                changes=changes,
                snapshot_status=snapshot_status,
            ),
        )
    return WorkspaceConflictError.from_reason(
        SERVER_CONFIG_DRIFT_REASON,
        diagnostics=WorkspaceDiagnostics(
            category="config_drift",
            reason=SERVER_CONFIG_DRIFT_REASON,
            changes=changes,
            snapshot_status=snapshot_status,
        ),
    )


# [해설][주의] 바인딩의 핵심 쓰기 경로. 하나의 SQLite 트랜잭션(BEGIN IMMEDIATE)에서
# [해설]   처리해 동시 바인딩 경쟁을 막는다.
# [해설][흐름] 1) 테이블 보장 후 INSERT OR IGNORE(첫 바인딩이면 삽입) 2) 새로 삽입됐을
# [해설]   때만 스냅샷 기록
# [해설][흐름] 3) 저장된 행을 다시 읽어 _binding_differs로 호환성 판정 -> 비호환이면
# [해설]   로그 후 WorkspaceConflictError
# [해설][흐름] 4) 옛 행이면 UPDATE로 현재 스키마로 이전(읽은 지문을 조건으로 걸어 동시
# [해설]   이전과의 덮어쓰기 방지)
# [해설][흐름] 5) 정책은 같고 런타임 지문만 바뀌었으면(모델 변경) 런타임 지문만 갱신 6)
# [해설]   그 외에는 기존 행을 그대로 반환.
def _bind(
    thread_id: str,
    proposed: WorkspaceBinding,
    snapshot: WorkspaceSnapshot,
) -> WorkspaceBinding:
    with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        _initialize(conn)
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO dcode_thread_workspaces (
                thread_id, schema_version, workspace_id, cwd, project_root,
                generation, resource_key, config_fingerprint,
                workspace_config_json, policy_fingerprint, runtime_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                proposed.schema_version,
                proposed.workspace_id,
                proposed.cwd,
                proposed.project_root,
                proposed.generation,
                proposed.resource_key,
                proposed.config_fingerprint,
                proposed.workspace_config_json,
                proposed.policy_fingerprint,
                proposed.runtime_fingerprint,
            ),
        )
        if cursor.rowcount:
            # Only a newly bound thread records a snapshot; a rebind of an
            # existing thread — including one about to be refused — must never
            # overwrite the evidence a later refusal diff reports from.
            _write_snapshot(conn, thread_id, snapshot)
        row = conn.execute(
            "SELECT * FROM dcode_thread_workspaces WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        if row is None:
            msg = f"workspace binding was not persisted for thread {thread_id}"
            raise RuntimeError(msg)
        existing = _row_binding(row)
        if _binding_differs(existing, proposed):
            conflict = _binding_conflict(
                thread_id, existing, proposed, _read_snapshot(conn, thread_id)
            )
            logger.warning(
                "Workspace binding refused for thread %s: %s",
                thread_id,
                conflict.diagnostics.log_summary()
                if conflict.diagnostics is not None
                else conflict,
            )
            raise conflict
        if _is_migratable(existing):
            # Guard on the fingerprint this transaction actually read, so a
            # concurrent migration cannot be overwritten after the fact. The
            # migration rewrites the row to the current schema with the new
            # policy/runtime fingerprints; conversation checkpoints and history
            # live in separate tables and are untouched.
            conn.execute(
                """
                UPDATE dcode_thread_workspaces
                SET schema_version = ?, resource_key = ?, config_fingerprint = ?,
                    workspace_config_json = ?, policy_fingerprint = ?,
                    runtime_fingerprint = ?
                WHERE thread_id = ? AND config_fingerprint = ?
                """,
                (
                    proposed.schema_version,
                    proposed.resource_key,
                    proposed.config_fingerprint,
                    proposed.workspace_config_json,
                    proposed.policy_fingerprint,
                    proposed.runtime_fingerprint,
                    thread_id,
                    existing.config_fingerprint,
                ),
            )
            return proposed
        # Policy-compatible rebind of a current row: refresh the full runtime
        # fingerprint when the runtime identity (model/params/prompt) changed,
        # so runtime validation accepts the new model and the runtime cache
        # rebuilds. The policy fingerprint and workspace identity are unchanged,
        # and the snapshot/comparison evidence is left intact. Guarded on the
        # policy fingerprint actually read so a concurrent policy change cannot
        # be overwritten after the fact.
        if (
            existing.policy_fingerprint
            and existing.policy_fingerprint == proposed.policy_fingerprint
            and existing.runtime_fingerprint != proposed.runtime_fingerprint
        ):
            conn.execute(
                """
                UPDATE dcode_thread_workspaces
                SET runtime_fingerprint = ?, config_fingerprint = ?
                WHERE thread_id = ? AND policy_fingerprint = ?
                """,
                (
                    proposed.runtime_fingerprint,
                    proposed.config_fingerprint,
                    thread_id,
                    existing.policy_fingerprint,
                ),
            )
            return proposed
        return existing


# [해설] 스레드 바인딩 읽기 전용 조회(없으면 None).
def _read(thread_id: str) -> WorkspaceBinding | None:
    with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        _initialize(conn)
        row = conn.execute(
            "SELECT * FROM dcode_thread_workspaces WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return _row_binding(row) if row is not None else None


# [해설] 새 스레드의 작업공간을 만들거나 기존 것과 대조한다. 동기
# [해설]   작업(resolve_workspace, _bind)은 스레드 풀로 보내
# [해설] 서버 이벤트 루프를 막지 않는다. 클라이언트 요청은 서버의 offload_api.py 쪽
# [해설]   라우트가 받아 이 함수를 부른다.
async def bind_thread_workspace(
    thread_id: str,
    cwd: object,
    workspace_config: object | None = None,
    *,
    config_fingerprint: str | None = None,
) -> WorkspaceBinding:
    """Atomically create or verify a thread workspace binding.

    Returns:
        The immutable binding for the thread.

    Raises:
        ValueError: If the thread is invalid.
    """
    if not isinstance(thread_id, str) or not thread_id:
        msg = "thread_id must be non-empty"
        raise ValueError(msg)
    proposed = await asyncio.to_thread(
        resolve_workspace,
        cwd,
        workspace_config,
        config_fingerprint=config_fingerprint,
    )
    snapshot = snapshot_for_payload(proposed.workspace_config())
    return await asyncio.to_thread(_bind, thread_id, proposed, snapshot)


# [해설] 스레드 바인딩 조회 비동기 래퍼. thread_id가 비정상이면 예외 대신 None.
async def get_thread_workspace(thread_id: str) -> WorkspaceBinding | None:
    """Read a thread's durable workspace binding.

    Returns:
        The binding, or `None` when the thread is unbound.
    """
    if not isinstance(thread_id, str) or not thread_id:
        return None
    return await asyncio.to_thread(_read, thread_id)


# [해설][주의] 실행(run) 요청 때마다 서버가 호출하는 검증 관문. server_graph.py의
# [해설]   make_graph가 execution.context의 workspace 값을 넘긴다.
# [해설][흐름] 1) payload가 비어 있지 않은 dict인지 확인 2) 클라이언트가 주장한 지문
# [해설]   결정(명시 지문 > workspace_config의 해시)
# [해설][흐름] 3) DB에서 행을 읽어 (a) 바인딩 없음 (b) 컨텍스트 불일치 (c) 지문 불일치
# [해설]   순으로 거부
# [해설][흐름] 4) 스키마 버전 확인 5) 저장된 cwd를 다시 resolve해 workspace_id가
# [해설]   그대로인지(디렉터리 신원 변조) 확인.
async def require_thread_workspace(
    thread_id: str,
    payload: object,
    workspace_config: object | None = None,
    *,
    config_fingerprint: str | None = None,
) -> WorkspaceBinding:
    """Validate run context against the durable workspace binding.

    Returns:
        The server-authoritative binding and persisted resource policy.

    Raises:
        TypeError: If workspace context is not an object.
        WorkspaceConflictError: If the context, policy, or workspace has changed.
    """
    if not isinstance(payload, dict) or not payload:
        msg = "workspace context is required"
        raise TypeError(msg)
    data = cast("dict[str, Any]", payload)
    # An explicit full `config_fingerprint` (the runtime identity) wins over
    # recomputing one from `workspace_config`: the payload omits model/runtime
    # fields, so its digest can never match a stored full fingerprint. Only
    # fall back to the payload digest when no explicit fingerprint is given.
    claimed_fingerprint = config_fingerprint
    if claimed_fingerprint is None and workspace_config is not None:
        _, claimed_fingerprint = canonical_workspace_config(workspace_config)

    def _require() -> WorkspaceBinding:
        with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            _initialize(conn)
            row = conn.execute(
                "SELECT * FROM dcode_thread_workspaces WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is None:
                msg = f"thread {thread_id} has no workspace binding"
                raise WorkspaceConflictError(
                    msg,
                    diagnostics=WorkspaceDiagnostics(
                        category="unbound_thread",
                        reason="no workspace binding",
                        snapshot_status="unavailable",
                    ),
                )
            existing = _row_binding(row)
            expected = existing.to_payload()
            if any(data.get(key) != value for key, value in expected.items()):
                msg = f"workspace context does not match thread {thread_id}"
                raise WorkspaceConflictError(
                    msg,
                    diagnostics=WorkspaceDiagnostics(
                        category="context_mismatch",
                        reason="workspace context does not match the binding",
                        snapshot_status=(
                            "current"
                            if _read_snapshot(conn, thread_id) is not None
                            else "unavailable"
                        ),
                    ),
                )
            if (
                claimed_fingerprint is not None
                and claimed_fingerprint != existing.config_fingerprint
            ):
                msg = f"workspace configuration does not match thread {thread_id}"
                raise WorkspaceConflictError(
                    msg,
                    diagnostics=WorkspaceDiagnostics(
                        category="fingerprint_mismatch",
                        reason="workspace configuration does not match the binding",
                        snapshot_status=(
                            "current"
                            if _read_snapshot(conn, thread_id) is not None
                            else "unavailable"
                        ),
                    ),
                )
            return existing

    existing = await asyncio.to_thread(_require)
    if existing.schema_version != _SCHEMA_VERSION:
        msg = f"workspace binding schema is unsupported for thread {thread_id}"
        raise WorkspaceConflictError(
            msg,
            diagnostics=WorkspaceDiagnostics(
                category="unsupported_schema",
                reason="workspace binding schema is unsupported",
                snapshot_status="unavailable",
                binding_schema_version=existing.schema_version,
                server_schema_version=_SCHEMA_VERSION,
            ),
        )
    resolved = await asyncio.to_thread(
        resolve_workspace,
        existing.cwd,
        existing.workspace_config(),
        config_fingerprint=existing.config_fingerprint,
    )
    if resolved.workspace_id != existing.workspace_id:
        msg = f"workspace identity changed for thread {thread_id}"
        raise WorkspaceConflictError(
            msg,
            diagnostics=WorkspaceDiagnostics(
                category="identity_changed",
                reason="workspace identity changed",
                snapshot_status="unavailable",
            ),
        )
    return existing
