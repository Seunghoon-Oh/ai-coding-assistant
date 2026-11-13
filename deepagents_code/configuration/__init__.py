"""Managed and user configuration: paths, providers, and merge.

Writes live in `deepagents_code.configuration.writer` and are imported from
there directly, so no writer symbol is re-exported here.
"""
# [해설] configuration 패키지의 공개 진입점이다. 이 파일은 로직이 없고, 하위 모듈에서
# [해설]   가져온 심볼을 `__all__`로 다시 내보내는 역할만 한다.
# [해설] 내보내는 심볼은 세 묶음이다. (1) 경로: `paths.py`의 `managed_config_path`,
# [해설]   `resolve_managed_path`, `ResolvedManagedPath`.
# [해설] (2) 관리형 정책 상태: `service.py`의 `get_config_sources`, `managed_health`,
# [해설]   `ManagedConfigError` 등 (이 파일 담당 레인 밖).
# [해설] (3) 기본 타입: `types.py`의 `ProviderHealth`, `ProviderStatus`.
# [해설] [설계] 쓰기 함수(`update_user_config` 등)는 일부러 여기서 내보내지 않는다.
# [해설]   쓰기는 `configuration/writer.py`에서 직접 import 하도록 해서, 읽기 API와 쓰기
# [해설]   API를 import 경로로 구분한다(모듈 docstring 참고).
# [해설] 패키지 전체 그림: 타입(`types.py`) -> 경로(`paths.py`) -> 제공자(`provider.py`,
# [해설]   `providers.py`) -> 병합(`resolver.py`) -> 스냅샷 서비스(`service.py`) ->
# [해설]   쓰기(`writer.py`) 순으로 쌓인다.

from deepagents_code.configuration.paths import (
    ResolvedManagedPath,
    managed_config_path,
    resolve_managed_path,
)
from deepagents_code.configuration.service import (
    ManagedConfigError,
    ManagedHealth,
    ManagedPolicyError,
    get_config_sources,
    invalidate_config_sources,
    managed_config_status,
    managed_declaration,
    managed_health,
    managed_policy_violations,
    require_healthy_managed_config,
)
from deepagents_code.configuration.types import ProviderHealth, ProviderStatus

__all__ = [
    "ManagedConfigError",
    "ManagedHealth",
    "ManagedPolicyError",
    "ProviderHealth",
    "ProviderStatus",
    "ResolvedManagedPath",
    "get_config_sources",
    "invalidate_config_sources",
    "managed_config_path",
    "managed_config_status",
    "managed_declaration",
    "managed_health",
    "managed_policy_violations",
    "require_healthy_managed_config",
    "resolve_managed_path",
]
