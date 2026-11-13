"""Pure ranked resolution and deep-merge logic for layered configuration.

The ranked engine is intentionally unaware of the manifest, UI, model, theme,
environment, or filesystem. Providers coerce their own domains before handing
`Found`, `Unset`, or `Invalid` results to this module. Human-readable source
labels likewise remain in `ProviderStatus`; provenance and health here use only
numeric ranks.
"""

# [해설] 이 모듈은 '설정 계층 해석기'의 순수 계산부다. 파일·환경변수를 직접 읽지 않고,
# [해설]   providers.py의 Provider들이 값을 타입에 맞게 변환(coerce)해 넘겨주면
# [해설]   순위(rank)로 승자를 고른다.
# [해설] 두 가지 해석 경로가 있다.
# [해설]   1) ConfigResolver / resolve_ranked: 옵션 하나를 provider 체인에서 해석
# [해설]   (매니페스트 옵션용).
# [해설]   2) merge_toml_tables: 관리형(managed) TOML 표를 사용자 TOML 표 위에
# [해설]   deep-merge (service.py가 사용).
# [해설] 핵심 규칙: 숫자가 작을수록 강하다. MANAGED(200) < CLI(300) < RELOAD(350) <
# [해설]   ENV(400) < USER(500) < DEFAULT(1000).
# [해설] 병합 전략은 옵션마다 replace / union / deep_merge 중 하나이며, 거부
# [해설]   목록(deny-list)은 union이라 약한 계층이
# [해설]   강한 계층의 금지 항목을 지울 수 없다(fail-closed).
# [해설] 구성 요소 연결: service.py(관리형 스냅샷 로드) -> get_config_resolver(캐시된
# [해설]   resolver 생성) -> ConfigResolver.get.
# [해설][주의] resolver는 numeric rank만 안다. 사람이 읽는 출처 이름(예:
# [해설]   'config.toml')은 ProviderStatus에 있다.
from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence
    from pathlib import Path

    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.configuration.provider import ConfigProvider
    from deepagents_code.configuration.types import TomlSnapshot

from deepagents_code.configuration.types import (
    Found,
    ProviderResult,
    ProviderStatus,
)

# [해설] 계층 순위표. 값이 작을수록 우선한다: 관리형 정책 200 > CLI 인자 300 > 런타임
# [해설]   reload 350 >
# [해설]   환경변수 400 > 사용자 config.toml 500 > 매니페스트 기본값 1000. 중간에 빈
# [해설]   번호를 둔 것은
# [해설]   나중에 계층을 끼워 넣기 위한 여유로 보이며 RELOAD_RANK 350이 그 틈에 실제로
# [해설]   끼어 있다(config.py:3688). 간격을 둔 의도를 적은 문서는 없다(추정).
# [해설]   RELOAD_RANK는 config.py가 쓴다.
MANAGED_RANK = 200
"""Managed policy rank; lower numeric ranks have stronger precedence."""

CLI_RANK = 300
"""Parsed command-line argument rank."""

RELOAD_RANK = 350
"""Retained runtime-reload value rank."""

ENVIRONMENT_RANK = 400
"""Process-environment rank."""

USER_RANK = 500
"""User `config.toml` rank."""

DEFAULT_RANK = 1000
"""Typed manifest-default rank."""


# [해설] Provider 하나가 옵션 하나에 대해 돌려주는 결과 묶음. result는
# [해설]   Found/Unset/Invalid 중 하나이고,
# [해설]   durable은 '프로세스가 끝나도 남는 출처인가'(파일=True, 환경변수·CLI=False)를
# [해설]   뜻해 masking 판단에 쓰인다.
@dataclass(frozen=True, slots=True)
class RankedProviderValue[T]:
    """One provider's already-coerced result for an option."""

    rank: int
    durable: bool
    status: ProviderStatus
    result: ProviderResult[T]
    diagnostics: tuple[str, ...] = ()
    """Ordered warnings encountered while trying aliases inside this tier."""


# [해설] 해석의 최종 결과. value 외에 '어느 계층이 기여했는가'(provenance)와 계층별
# [해설]   상태를 함께 들고 있어
# [해설]   `dcode config` 같은 화면이 출처 열을 그릴 수 있다. __post_init__이 계층 간
# [해설]   일관성을 생성 시점에 검증한다.
@dataclass(frozen=True, slots=True)
class ResolvedValue[T]:
    """Resolved value with rank-keyed provenance and provider health.

    Six of the seven fields are parallel rank-keyed collections whose mutual
    consistency is the entire meaning of the type, and consumers index straight
    into them: `config_manifest._ranked_source` reads
    `provider_status[rank] for rank in ranks` to render the source column, so
    an inconsistent instance is a `KeyError` in user-facing output. The
    invariants are checked at construction rather than documented, following
    `TomlSnapshot` in the same package.
    """

    value: T
    provenance: Mapping[int, frozenset[tuple[str, ...]]]
    tier_health: Mapping[int, ProviderResult[T]]
    provider_status: Mapping[int, ProviderStatus]
    masked_ranks: frozenset[int] = frozenset()
    selected_ranks: tuple[int, ...] = ()
    tier_diagnostics: Mapping[int, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Reject a value whose rank-keyed halves disagree.

        Also copies each mapping behind a `MappingProxyType`. `frozen=True`
        protects the field bindings, not the contents: without the copy a
        caller keeps a live reference to a dict this type presents as a
        read-only snapshot.

        Raises:
            ValueError: If a contributing rank is missing provider status, or
                is also reported as masked.
        """
        for name in (
            "provenance",
            "tier_health",
            "provider_status",
            "tier_diagnostics",
        ):
            frozen = MappingProxyType(dict(getattr(self, name)))
            object.__setattr__(self, name, frozen)
        # Both halves of `ranks`, not just `selected_ranks`: it falls back to
        # `provenance` when no rank was selected, and `_ranked_source` indexes
        # `provider_status` by whichever half it gets. Validating one branch
        # left the documented failure mode reachable through the other.
        contributing = set(self.selected_ranks) | set(self.provenance)
        missing = contributing - self.provider_status.keys()
        if missing:
            msg = (
                f"contributing ranks {sorted(missing)} have no provider "
                "status; rendering provenance would raise KeyError"
            )
            raise ValueError(msg)
        both = self.masked_ranks & set(self.selected_ranks)
        if both:
            msg = f"ranks {sorted(both)} cannot be both selected and masked"
            raise ValueError(msg)

    @property
    def ranks(self) -> tuple[int, ...]:
        """Contributing ranks in precedence order."""
        return self.selected_ranks or tuple(sorted(self.provenance))


type ConfigKey = str
"""Canonical dotted manifest key."""


# [해설] provider 체인(순위순)을 보관하고 옵션을 해석하는 클래스. 모든 읽기·갱신이 RLock
# [해설]   하나 아래에서 일어나
# [해설]   한 번의 해석이 '한 세대(generation)'의 provider만 본다는 보장을 준다.
class ConfigResolver:
    """Resolve manifest options through an ordered provider chain."""

    def __init__(self, providers: Sequence[ConfigProvider]) -> None:
        """Build a resolver with providers sorted by precedence.

        Args:
            providers: Configuration providers with unique numeric ranks.

        Raises:
            ValueError: If two providers declare the same rank.
        """
        ordered = tuple(sorted(providers, key=lambda provider: provider.rank))
        ranks = tuple(provider.rank for provider in ordered)
        if len(set(ranks)) != len(ranks):
            msg = "config providers must have unique ranks"
            raise ValueError(msg)
        self._providers = ordered
        self._lock = threading.RLock()

    def get[T](self, option: ConfigOption[T]) -> ResolvedValue[T]:
        """Resolve one option through every provider.

        Args:
            option: Manifest option to resolve.

        Returns:
            Resolved value with rank-keyed provenance and health.
        """
        with self._lock:
            return self._resolve(option, self._providers)

    def get_without_ranks[T](
        self, option: ConfigOption[T], ranks: Collection[int]
    ) -> ResolvedValue[T]:
        """Resolve one option after excluding selected provider ranks.

        Args:
            option: Manifest option to resolve.
            ranks: Provider ranks to omit from this read.

        Returns:
            Resolved value from the remaining providers.
        """
        with self._lock:
            providers = tuple(
                provider for provider in self._providers if provider.rank not in ranks
            )
            return self._resolve(option, providers)

    @staticmethod
    # [해설][흐름] 옵션 하나의 해석 순서: 1) 모든 provider에서 값을 읽는다 2)
    # [해설]   union/deep_merge 전략이면 기본값 계층을 제외하고
    # [해설]   resolve_ranked에 넘긴다 3) 아무도 Found가 아니면 DefaultProvider 값으로
    # [해설]   마지막 시도를 한다.
    def _resolve[T](
        option: ConfigOption[T],
        providers: Sequence[ConfigProvider],
    ) -> ResolvedValue[T]:
        """Resolve one option against a lock-held provider generation.

        Args:
            option: Manifest option to resolve.
            providers: Providers frozen to one generation.

        Returns:
            Resolved value with rank-keyed provenance and health.

        Raises:
            RuntimeError: If no provider returns a value.
        """
        values = tuple(provider.get(option) for provider in providers)
        strategy = option.merge_strategy.value
        effective_values = (
            tuple(value for value in values if value.rank != DEFAULT_RANK)
            if strategy in {"union", "deep_merge"}
            else values
        )
        resolved = resolve_ranked(effective_values, strategy=strategy)
        if resolved is None:
            from deepagents_code.configuration.providers import DefaultProvider

            fallback = DefaultProvider().get(option)
            without_default = tuple(
                value for value in values if value.rank != DEFAULT_RANK
            )
            resolved = resolve_ranked(
                (*without_default, fallback),
                strategy=strategy,
            )
        if resolved is None:
            msg = f"fallback provider was unset for {option.key}"
            raise RuntimeError(msg)
        return resolved

    def resolve_options(
        self,
        options: Sequence[ConfigOption[object]],
    ) -> Mapping[ConfigKey, ResolvedValue[object]]:
        """Resolve selected options against one provider generation.

        Resolving an option is not uniformly cheap: `THEME_DELEGATE` reaches
        the theme registry, which imports Textual (~470ms). Callers on the
        startup hot path ask for the options they need rather than the whole
        manifest, and keep the single-generation guarantee either way.

        Args:
            options: Manifest options to resolve together.

        Returns:
            Immutable mapping from canonical option key to resolved value.
        """
        with self._lock:
            resolved = {
                option.key: self._resolve(option, self._providers) for option in options
            }
        return MappingProxyType(resolved)

    def resolve_all(self) -> Mapping[ConfigKey, ResolvedValue[object]]:
        """Resolve the full manifest against one provider generation.

        Returns:
            Immutable mapping from canonical option key to resolved value.
        """
        from deepagents_code.config_manifest import get_config_options

        return self.resolve_options(get_config_options())

    # [해설][주의] reload는 resolver 락을 잡은 채 provider를 다시 읽는다. 원격 관리형
    # [해설]   설정이면 네트워크 대기 동안
    # [해설]   다른 get 호출이 모두 막혀 UI가 멈출 수 있다. 그래서
    # [해설]   get_config_resolver(refresh_managed=True) 경로를 쓴다.
    def reload(self) -> None:
        """Propagate a source refresh to every provider.

        Advances the generation, so this also re-arms the source-level
        diagnostics -- see `reload_with_replacements`, which does the work.

        Reloads every provider *under this resolver's lock*, and a managed
        provider's loader can fetch over the network for the whole remote
        timeout. Every concurrent `get` and `resolve_options` blocks for that
        duration, which on the Textual event loop stalls the UI. Callers that
        may run against a remote descriptor should fetch first and hand the
        snapshot over -- `get_config_resolver(refresh_managed=True)` does
        exactly that -- rather than calling this.
        """
        self.reload_with_replacements({})

    # [해설] 이미 새 세대로 갱신해 둔 provider(주로 관리형)를 교체 설치하고 나머지만
    # [해설]   reload한다.
    # [해설]   사용 불가한 교체본은 거부한다: 그 계층이 '아무것도 선언하지 않음'으로
    # [해설]   읽혀 약한 계층이 이기는 것을 막기 위함.
    def reload_with_replacements(
        self,
        replacements: Mapping[int, ConfigProvider],
    ) -> None:
        """Refresh providers after installing already-refreshed replacements.

        Non-replaced providers are reloaded under this resolver's lock, so the
        caveat on `reload` about a managed loader's network I/O applies to any
        call that leaves the managed rank unreplaced.

        Args:
            replacements: Providers already bound to the desired generation,
                keyed by the rank they replace. Replacements are installed but
                not reloaded, so each must already hold a usable snapshot.

        Raises:
            ValueError: If a replacement rank is not in this resolver, or a
                replacement is not serving a usable snapshot.
        """
        with self._lock:
            ranks = {provider.rank for provider in self._providers}
            unknown = replacements.keys() - ranks
            if unknown:
                msg = f"cannot replace unknown provider ranks: {sorted(unknown)}"
                raise ValueError(msg)
            unusable = sorted(
                rank
                for rank, provider in replacements.items()
                if not _serves_usable_policy(provider)
            )
            if unusable:
                msg = (
                    f"replacement providers at ranks {unusable} are unusable; "
                    "installing one would drop the source's restrictions"
                )
                raise ValueError(msg)
            self._providers = tuple(
                replacements.get(provider.rank, provider)
                for provider in self._providers
            )
            for provider in self._providers:
                if provider.rank not in replacements:
                    provider.reload()

        # The source-level diagnostics (`shadowed`, `unusable`, `retained`) are
        # deduplicated so one `dcode config` sweep over ~200 options does not
        # print the same rejection 200 times. Scoping that to the process made
        # the second `/reload` of a still-broken file silent -- the reason
        # string is identical, so the user gets no message during exactly the
        # edit-and-retry loop where they need one. A generation advance ends
        # the sweep the dedup exists for, so it is the right scope.
        from deepagents_code.config_manifest import reset_source_diagnostics

        reset_source_diagnostics()

    def provider_statuses(self) -> Mapping[int, ProviderStatus]:
        """Return immutable provider health keyed by precedence rank."""
        with self._lock:
            statuses = {
                provider.rank: provider.status() for provider in self._providers
            }
        return MappingProxyType(statuses)

    # [해설] CLI 계층은 argparse 이후에 생기므로, 이미 만들어진 resolver에 순위 정렬을
    # [해설]   유지하며 끼워 넣는다.
    def install_provider(self, provider: ConfigProvider) -> None:
        """Insert a provider into the live chain, keeping rank order.

        Used for the CLI tier, which exists only after `argparse` runs — long
        after this resolver may have been built and cached. Unlike
        `reload_with_replacements`, this advances no generation and touches no
        files: the CLI provider is in-memory, so there is nothing to reload,
        and re-arming source diagnostics for an install that invalidates no
        snapshot would only risk a duplicate warning on the next resolution.

        Args:
            provider: Provider to insert. Its rank must not already be present.

        Raises:
            ValueError: If a provider already serves the new provider's rank.
        """
        with self._lock:
            ranks = {existing.rank for existing in self._providers}
            if provider.rank in ranks:
                msg = f"a provider already serves rank {provider.rank}"
                raise ValueError(msg)
            self._providers = tuple(
                sorted((*self._providers, provider), key=lambda p: p.rank)
            )

    def toml_snapshot(self, rank: int) -> TomlSnapshot | None:
        """Return the cached TOML snapshot at `rank`, if that provider is one.

        Lets a caller build a one-off resolver against the same file
        generation this resolver is serving -- for example, re-resolving an
        option with the managed tier masked while keeping the shared user
        snapshot instead of re-parsing `config.toml` off disk.

        Propagates the `RuntimeError` `current_snapshot` raises when the
        provider at `rank` produced no snapshot.

        Args:
            rank: Precedence rank whose snapshot to return.

        Returns:
            The provider's current snapshot, or `None` when no TOML provider
            sits at `rank` (environment and default providers carry none).
        """
        from deepagents_code.configuration.providers import TomlFileProvider

        with self._lock:
            for provider in self._providers:
                if provider.rank == rank and isinstance(provider, TomlFileProvider):
                    return provider.current_snapshot()
        return None


# [해설] 표준 provider 체인을 조립하는 곳. 순서는 managed(200) -> (있으면) CLI(300) ->
# [해설]   Env(400) -> user(500) -> Default(1000).
# [해설]   managed/user 스냅샷이 같은 타입이라 서로 바꿔 넣으면 사용자 파일이 관리형
# [해설]   권한을 얻으므로 키워드 전용이다.
def resolver_from_snapshots(
    *,
    managed: TomlSnapshot,
    user: TomlSnapshot,
    managed_loader: Callable[[], TomlSnapshot] | None = None,
    user_loader: Callable[[], TomlSnapshot] | None = None,
    cli_provider: ConfigProvider | None = None,
) -> ConfigResolver:
    """Build the standard provider chain from one file-snapshot generation.

    Keyword-only by design. The two snapshots share a type, so a positional
    transposition would load the user's writable `config.toml` at
    `MANAGED_RANK` -- user data acquiring managed precedence, which is the one
    escalation this trust boundary exists to prevent. Nothing downstream
    rejects it: this function never inspects `managed.status.name`, and
    `TomlFileProvider` accepts whatever rank it is handed.

    Args:
        managed: Managed TOML snapshot.
        user: User TOML snapshot.
        managed_loader: Optional managed reload operation.
        user_loader: Optional user reload operation.
        cli_provider: Optional parsed-argument provider for this process.

    Returns:
        Resolver containing managed, environment, user, and default providers,
            plus the CLI provider when one is supplied.
    """
    from deepagents_code.configuration.providers import (
        DefaultProvider,
        EnvProvider,
        TomlFileProvider,
    )

    # Snapshots built in-memory carry no path. Pass that through rather than
    # inventing a relative filename: `TomlFileProvider.load` would resolve it
    # against the process working directory, so a later `reload()` on a
    # diagnostic resolver would read whatever `./managed_config.toml` happens
    # to sit in the repo the agent is running in and treat it as policy.
    managed_path = managed.status.path
    user_path = user.status.path
    # [해설][설계] 여기 적힌 튜플 순서가 곧 우선순위의 근거다. ConfigResolver.__init__이
    # [해설]   rank로 다시 정렬하므로 순서는 읽기 편의용이다.
    providers: tuple[ConfigProvider, ...] = (
        TomlFileProvider(
            name=managed.status.name,
            path=managed_path,
            rank=MANAGED_RANK,
            durable=True,
            snapshot=managed,
            loader=managed_loader,
        ),
        *((cli_provider,) if cli_provider is not None else ()),
        EnvProvider(),
        TomlFileProvider(
            name=user.status.name,
            path=user_path,
            rank=USER_RANK,
            durable=True,
            snapshot=user,
            loader=user_loader,
        ),
        DefaultProvider(),
    )
    return ConfigResolver(providers)


# [해설] 공유 resolver 캐시의 키: (사용자 config 경로, 관리형 파일 경로). 두 경로 중
# [해설]   하나라도 바뀌면 resolver를 새로 만든다.
@dataclass(frozen=True, slots=True)
class _ResolverKey:
    """The pair of file paths a shared resolver is built for.

    Named rather than a bare `tuple[object, ...]`: a key built with different
    arity or field order compares unequal, silently rebuilds the resolver, and
    loses the single-generation guarantee with no error anywhere.
    """

    user_path: Path
    managed_path: Path | None


# [해설] 프로세스 전역 resolver 캐시. entry(키+resolver)와 cli_provider를 한 곳에 두어
# [해설]   서로 어긋난 상태를 막는다.
@dataclass(slots=True)
class _ResolverCache:
    """Mutable process resolver cache guarded by one lifecycle lock.

    One field, not a key and a resolver side by side: those admit a populated
    key with no resolver, and a lookup that trusts either half alone would then
    read a stale generation or rebuild one that already exists.
    """

    entry: tuple[_ResolverKey, ConfigResolver] | None = None
    cli_provider: ConfigProvider | None = None


_resolver_cache_lock = threading.RLock()
_resolver_cache = _ResolverCache()


def installed_cli_provider() -> ConfigProvider | None:
    """Return the parsed-argument provider installed for this process.

    Ad-hoc resolvers built from caller-supplied snapshots do not go through the
    process cache, so they have no CLI tier unless they ask for this one. A
    reader that omits it reports the wrong source for any option a flag in the
    current argv is setting.

    Returns:
        The installed provider, or `None` before `install_cli_provider` runs.
    """
    with _resolver_cache_lock:
        return _resolver_cache.cli_provider


# [해설] 관리형 스냅샷을 새로 읽되, 정책 위반이 있으면 새 것을 버리고 직전 캐시를
# [해설]   유지한다(관리자 정책 보호).
def _reload_enforceable_managed_snapshot() -> TomlSnapshot:
    """Return a refreshed managed snapshot only when policy can enforce it."""
    from deepagents_code.configuration.service import (
        get_managed_snapshot,
        managed_policy_violations,
    )

    candidate = get_managed_snapshot(refresh=True)
    if candidate.status.usable and managed_policy_violations(
        candidate.data,
        status=candidate.status,
    ):
        return get_managed_snapshot()
    return candidate


# [해설] 프로세스 공유 resolver를 돌려주는 단일 진입점. 설정 읽기는 거의 모두 여기를
# [해설]   거친다.
# [해설][흐름] 1) 관리형 스냅샷 결정(인자 > refresh > 캐시) 2) 캐시 키 계산 3) 캐시
# [해설]   미스면 user TOML을 읽어
# [해설]   resolver_from_snapshots로 새로 조립 4) 캐시 히트이면 refresh_managed일 때만
# [해설]   관리형 provider를 교체 후 reload.
def get_config_resolver(
    *,
    refresh_managed: bool = False,
    managed_snapshot: TomlSnapshot | None = None,
    cli_provider: ConfigProvider | None = None,
) -> ConfigResolver:
    """Return the shared process resolver for the active config paths.

    Args:
        refresh_managed: Refresh the user and environment tiers on an existing
            matching resolver. The managed tier is not re-read: it is replaced
            with the snapshot the caller already validated, so one reload
            observes one managed-file generation.
        managed_snapshot: Already refreshed and validated managed snapshot.
            Supplies the cache key, and builds the resolver when the cache
            misses. On a cache hit it is installed only when `refresh_managed`
            is set -- without it the resolver keeps the generation it is
            already serving, so the snapshot must be that same generation.
        cli_provider: Parsed-argument provider to install for this process.

    Returns:
        Resolver shared by consumers of the active managed and user paths.

    Raises:
        ValueError: If `managed_snapshot` is a different generation than the
            one already installed and `refresh_managed` is not set. The
            snapshot would otherwise be discarded in silence. Also if
            `cli_provider` differs from the one already installed for this
            process: one argv yields one CLI tier, and silently keeping either
            provider would misreport every flag the other one carries.
    """
    from deepagents_code.configuration.providers import TomlFileProvider
    from deepagents_code.configuration.service import get_managed_snapshot
    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    # [해설][흐름] 1) 어떤 관리형 스냅샷을 쓸지 정한다: 호출자가 준 것 > 강제 갱신 결과
    # [해설]   > 캐시된 것.
    if managed_snapshot is not None:
        managed = managed_snapshot
    elif refresh_managed:
        managed = _reload_enforceable_managed_snapshot()
    else:
        managed = get_managed_snapshot()
    key = _ResolverKey(DEFAULT_CONFIG_PATH, managed.status.path)
    # [해설][흐름] 2) 락 안에서 CLI provider를 기록하고, 캐시가 없거나 키가 다르거나 CLI
    # [해설]   계층이 빠져 있으면 새로 만든다.
    with _resolver_cache_lock:
        installed_cli = _resolver_cache.cli_provider
        if cli_provider is not None:
            if installed_cli is not None and installed_cli != cli_provider:
                msg = "a different CLI provider is already installed for this process"
                raise ValueError(msg)
            _resolver_cache.cli_provider = cli_provider
            installed_cli = cli_provider
        entry = _resolver_cache.entry
        if (
            entry is None
            or entry[0] != key
            or (
                cli_provider is not None
                and CLI_RANK not in entry[1].provider_statuses()
            )
        ):
            # [해설][흐름] 3) 캐시 미스: 사용자 config.toml을 읽고 표준 체인을 조립해
            # [해설]   캐시에 저장한다.
            user_provider = TomlFileProvider(
                name="config.toml", path=DEFAULT_CONFIG_PATH
            )
            user = user_provider.load()
            resolver = resolver_from_snapshots(
                managed=managed,
                user=user,
                managed_loader=_reload_enforceable_managed_snapshot,
                user_loader=user_provider.load,
                cli_provider=installed_cli,
            )
            _resolver_cache.entry = (key, resolver)
            # A rebuild is a generation advance too: the key changes when
            # managed policy is installed or removed, and the dedup set would
            # otherwise carry rejections from the generation just replaced.
            from deepagents_code.config_manifest import reset_source_diagnostics

            reset_source_diagnostics()
            return resolver
        resolver = entry[1]
        if not refresh_managed:
            # A caller that hands over a snapshot and does not ask for a
            # refresh is telling the resolver to keep serving what it has, so
            # the two must already agree. They do today -- the preview path
            # takes its snapshot with `refresh=False`, which returns the
            # cached one -- but nothing in the signature says so, and the
            # alternative is discarding a validated generation in silence.
            if managed_snapshot is not None:
                installed = resolver.toml_snapshot(MANAGED_RANK)
                if installed is not None and installed != managed_snapshot:
                    msg = (
                        "managed_snapshot is a different generation than the "
                        "one in force; pass refresh_managed=True to install it"
                    )
                    raise ValueError(msg)
            return resolver
        # [해설][흐름] 4) 캐시 히트 + refresh_managed: 관리형 계층만 새 스냅샷
        # [해설]   provider로 교체하고 나머지는 reload한다.
        resolver.reload_with_replacements(
            {
                MANAGED_RANK: _managed_replacement_provider(
                    resolver,
                    managed,
                )
            }
        )
        return resolver


def _serves_usable_policy(provider: ConfigProvider) -> bool:
    """Whether a replacement is serving a snapshot resolution can trust.

    A replacement bypasses `reload`, so it needs a usable generation behind it.
    Its latest status may still be unusable while it safely retains the
    previous snapshot; rejecting that state would erase the failed-refresh
    diagnostic. A provider whose *served* snapshot is unusable would resolve as
    "this source declares nothing" and silently let lower ranks win.

    Args:
        provider: Replacement about to be installed.

    Returns:
        Whether the generation this provider resolves from is usable.
    """
    from deepagents_code.configuration.providers import TomlFileProvider

    if provider.status().usable:
        return True
    return (
        isinstance(provider, TomlFileProvider)
        and provider.current_snapshot().status.usable
    )


def _managed_replacement_provider(
    resolver: ConfigResolver,
    candidate: TomlSnapshot,
) -> ConfigProvider:
    """Build a current managed replacement that retains failed refreshes safely.

    Args:
        resolver: Resolver currently serving the previous generation.
        candidate: Snapshot fetched before taking the resolver lock. A newer
            enforceable generation may have published while this caller waited.

    Returns:
        Replacement carrying the current enforceable generation or the
        candidate's failed-refresh status.

    Raises:
        RuntimeError: If the shared resolver has no managed TOML provider.
    """
    from deepagents_code.configuration.providers import TomlFileProvider
    from deepagents_code.configuration.service import get_managed_snapshot

    installed = resolver.toml_snapshot(MANAGED_RANK)
    if installed is None:
        msg = "shared config resolver has no managed TOML provider"
        raise RuntimeError(msg)
    if candidate.status.usable:
        candidate = get_managed_snapshot()
    replacement = TomlFileProvider(
        name=candidate.status.name,
        path=candidate.status.path,
        rank=MANAGED_RANK,
        durable=True,
        snapshot=installed,
        loader=_reload_enforceable_managed_snapshot,
    )
    replacement.reload_from_snapshot(candidate)
    return replacement


# [해설] CLI provider를 설정 파일을 읽지 않고 설치한다. `--help` 같은 빠른 경로의 시작
# [해설]   시간을 지키기 위해
# [해설]   이미 캐시된 resolver에는 붙이고, 아직 없으면 보관했다가 첫
# [해설]   get_config_resolver가 가져가게 한다.
def install_cli_provider(cli_provider: ConfigProvider) -> None:
    """Install the process CLI provider without touching config files.

    Unlike `get_config_resolver(cli_provider=...)`, this never imports
    `deepagents_code.model_config` and never reads a TOML snapshot: it either
    attaches the provider to the already-cached resolver or stashes it for the
    first real `get_config_resolver` call to pick up. The startup fast paths
    (`--help`, bare command groups) parse arguments and return before any
    config resolution happens, so paying the settings-bootstrap import cost
    here would break the startup-perf contract those paths are tested
    against.

    Args:
        cli_provider: Parsed-argument provider to install for this process.

    Raises:
        ValueError: If a different CLI provider is already installed.
    """
    with _resolver_cache_lock:
        installed = _resolver_cache.cli_provider
        if installed is not None and installed != cli_provider:
            msg = "a different CLI provider is already installed for this process"
            raise ValueError(msg)
        _resolver_cache.cli_provider = cli_provider
        entry = _resolver_cache.entry
        if entry is not None and CLI_RANK not in entry[1].provider_statuses():
            entry[1].install_provider(cli_provider)


def reset_config_resolver() -> None:
    """Drop the cached process resolver.

    Test-only, and paired with `service.invalidate_config_sources`: the two
    caches are keyed differently, so clearing only the managed snapshot leaves
    this one serving the previous test's generation. Tests escaped that today
    only by incidentally monkeypatching `DEFAULT_CONFIG_PATH`, which changes
    the key; one that exercises the resolver at an unchanged path would inherit
    stale state.
    """
    # No `reset_source_diagnostics` here: dropping the entry makes the next
    # `get_config_resolver` take the cache-miss branch, which re-arms them as
    # part of building the new generation. Importing the manifest from this
    # teardown path also breaks the test that stubs it out of `sys.modules`.
    with _resolver_cache_lock:
        _resolver_cache.entry = None
        _resolver_cache.cli_provider = None


# [해설] 이 모듈의 핵심. 순위가 낮은(강한) Found부터 보되, 전략에 따라 승자
# [해설]   하나(replace)나 합산(union/deep_merge)을 만든다.
# [해설][주의] replace 전략의 masking은 한 방향이다: durable(파일) 값은 더 약한
# [해설]   비영속(환경·CLI 등) 값을 가리지만,
# [해설]   약한 계층의 값이 더 강한 환경변수를 가릴 수는 없다. 예: USER(500) 값은
# [해설]   ENV(400) 값을 못 이긴다.
def resolve_ranked[T](
    providers: Sequence[RankedProviderValue[T]],
    *,
    strategy: str = "replace",
) -> ResolvedValue[T] | None:
    """Resolve provider results by numeric rank and per-option merge strategy.

    Lower ranks win. For replacement options, a `Found` from a durable tier
    masks lower-precedence non-durable tiers. The mask is intentionally
    directional: a persisted user value at rank 500 cannot retroactively hide
    a higher-precedence environment value at rank 400.

    Accumulating strategies combine tiers by definition, so they retain every
    valid contribution. This preserves the existing fail-closed deny-list
    unions and deep TOML composition; treating accumulation as replacement
    would silently discard restrictions or sibling table leaves.

    Args:
        providers: Already-coerced provider results. Ranks must be unique.
        strategy: `replace`, `union`, or `deep_merge`.

    Returns:
        A resolved value, or `None` when no provider returned `Found`.

    Raises:
        ValueError: If ranks repeat or `strategy` is unknown.
    """
    # [해설][흐름] 1) 순위로 정렬하고 중복 순위·미지의 전략을 거부 2) 계층별 상태/진단
    # [해설]   맵 작성 3) Found가 없으면 None
    # [해설]   4) union/deep_merge는 전용 함수로 위임 5) replace는 masking을 계산한 뒤
    # [해설]   첫 비-masked Found를 승자로 정한다.
    ordered = sorted(providers, key=lambda provider: provider.rank)
    ranks = [provider.rank for provider in ordered]
    if len(set(ranks)) != len(ranks):
        msg = "ranked config providers must have unique ranks"
        raise ValueError(msg)
    if strategy not in {"replace", "union", "deep_merge"}:
        msg = f"unknown config merge strategy: {strategy}"
        raise ValueError(msg)

    tier_health = MappingProxyType(
        {provider.rank: provider.result for provider in ordered}
    )
    provider_status = MappingProxyType(
        {provider.rank: provider.status for provider in ordered}
    )
    tier_diagnostics = MappingProxyType(
        {provider.rank: provider.diagnostics for provider in ordered}
    )
    found = [provider for provider in ordered if isinstance(provider.result, Found)]
    if not found:
        return None
    if strategy == "union":
        return _resolve_ranked_union(
            found,
            tier_health,
            provider_status,
            tier_diagnostics,
        )
    if strategy == "deep_merge":
        return _resolve_ranked_deep_merge(
            found,
            tier_health,
            provider_status,
            tier_diagnostics,
        )

    durable_ranks = tuple(provider.rank for provider in found if provider.durable)
    masked = frozenset(
        provider.rank
        for provider in found
        if not provider.durable
        and any(durable_rank < provider.rank for durable_rank in durable_ranks)
    )
    winner = next(provider for provider in found if provider.rank not in masked)
    return ResolvedValue(
        _provider_value(winner),
        MappingProxyType({winner.rank: frozenset({()})}),
        tier_health,
        provider_status,
        masked,
        (winner.rank,),
        tier_diagnostics,
    )


def _replace_with_strongest[T](
    found: Sequence[RankedProviderValue[T]],
    tier_health: Mapping[int, ProviderResult[T]],
    provider_status: Mapping[int, ProviderStatus],
    tier_diagnostics: Mapping[int, tuple[str, ...]],
) -> ResolvedValue[T]:
    """Resolve to the strongest-precedence provider when accumulation fails.

    The value is copied. Provider values alias the process-wide managed
    snapshot, so handing out a live reference would let a consumer mutate
    administrator policy for the rest of the session.

    Returns:
        The lowest-rank provider's value, deep-copied.
    """
    winner = found[0]
    return ResolvedValue(
        deepcopy(_provider_value(winner)),
        MappingProxyType({winner.rank: frozenset({()})}),
        tier_health,
        provider_status,
        selected_ranks=(winner.rank,),
        tier_diagnostics=tier_diagnostics,
    )


# [해설] 목록형 값을 약한 계층부터 강한 계층 순으로 합친다. 한 계층이라도 목록이 아니면
# [해설]   가장 강한 값으로 대체한다.
def _resolve_ranked_union[T](
    found: Sequence[RankedProviderValue[T]],
    tier_health: Mapping[int, ProviderResult[T]],
    provider_status: Mapping[int, ProviderStatus],
    tier_diagnostics: Mapping[int, tuple[str, ...]],
) -> ResolvedValue[T]:
    """Accumulate list-like providers from weakest to strongest rank.

    Returns:
        The union, or the strongest-precedence (lowest-rank) replacement when a
        value is not list-like.
    """
    entries = [union_entries(_provider_value(provider)) for provider in found]
    if any(value is None for value in entries):
        return _replace_with_strongest(
            found, tier_health, provider_status, tier_diagnostics
        )
    union: list[Any] = []
    for value in reversed(entries):
        union = union_lists(union, cast("list[Any]", value))
    provenance = MappingProxyType(
        {provider.rank: frozenset({()}) for provider in found}
    )
    return ResolvedValue(
        cast("T", union),
        provenance,
        tier_health,
        provider_status,
        selected_ranks=tuple(provider.rank for provider in found),
        tier_diagnostics=tier_diagnostics,
    )


# [해설] 표(table) 값을 약한 계층에서 강한 계층으로 deep-merge하며 leaf별로 어느 rank가
# [해설]   기여했는지 provenance를 쌓는다.
def _resolve_ranked_deep_merge[T](
    found: Sequence[RankedProviderValue[T]],
    tier_health: Mapping[int, ProviderResult[T]],
    provider_status: Mapping[int, ProviderStatus],
    tier_diagnostics: Mapping[int, tuple[str, ...]],
) -> ResolvedValue[T]:
    """Deep-merge mapping providers from weakest to strongest rank.

    A tier that does not hold a mapping cannot be merged. Such a tier falls
    back to replacement by the strongest-precedence (lowest-rank) provider,
    matching `_resolve_ranked_union`. Returning the non-mapping tier itself
    would let a weaker tier displace managed policy.

    Returns:
        The merged mapping, or the strongest provider's value when any tier
        cannot be merged.
    """
    weakest = found[-1]
    value = _provider_value(weakest)
    if not isinstance(value, dict):
        return _replace_with_strongest(
            found, tier_health, provider_status, tier_diagnostics
        )
    merged = deepcopy(cast("dict[str, Any]", value))
    leaves = _ranked_leaf_provenance(merged, weakest.rank)
    for provider in reversed(found[:-1]):
        higher = _provider_value(provider)
        if not isinstance(higher, dict):
            return _replace_with_strongest(
                found, tier_health, provider_status, tier_diagnostics
            )
        merged, leaves = _merge_ranked_tables(
            merged,
            cast("dict[str, Any]", higher),
            leaves,
            provider.rank,
        )
    grouped: dict[int, set[tuple[str, ...]]] = {}
    for path, rank in leaves.items():
        grouped.setdefault(rank, set()).add(path)
    provenance = MappingProxyType(
        {rank: frozenset(paths) for rank, paths in grouped.items()}
    )
    return ResolvedValue(
        cast("T", merged),
        provenance,
        tier_health,
        provider_status,
        selected_ranks=tuple(provider.rank for provider in found),
        tier_diagnostics=tier_diagnostics,
    )


# [해설] 두 표를 재귀 병합한다. 양쪽이 표면 안으로 내려가고, 한쪽이 스칼라면 높은 쪽이
# [해설]   통째로 대체하며 그 하위 provenance를 지운다.
def _merge_ranked_tables(
    lower: dict[str, Any],
    higher: dict[str, Any],
    provenance: dict[tuple[str, ...], int],
    higher_rank: int,
    *,
    prefix: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[tuple[str, ...], int]]:
    """Deep-merge two mappings while retaining tuple-path rank provenance.

    Returns:
        The merged table and tuple-path-to-rank provenance.
    """
    merged = deepcopy(lower)
    ranked = dict(provenance)
    for key, value in higher.items():
        path = (*prefix, key)
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key], ranked = _merge_ranked_tables(
                cast("dict[str, Any]", existing),
                cast("dict[str, Any]", value),
                ranked,
                higher_rank,
                prefix=path,
            )
            continue
        merged[key] = deepcopy(value)
        for leaf in tuple(ranked):
            if leaf[: len(path)] == path:
                ranked.pop(leaf)
        ranked.update(_ranked_leaf_provenance(value, higher_rank, path))
    return merged, ranked


def _ranked_leaf_provenance(
    value: object, rank: int, path: tuple[str, ...] = ()
) -> dict[tuple[str, ...], int]:
    """Attribute every leaf under `value` to a numeric provider rank.

    Returns:
        Tuple-path-to-rank provenance for every leaf.
    """
    if isinstance(value, dict):
        if not value:
            return {path: rank} if path else {}
        result: dict[tuple[str, ...], int] = {}
        for key, child in cast("dict[str, object]", value).items():
            result.update(_ranked_leaf_provenance(child, rank, (*path, key)))
        return result
    return {path: rank}


def _provider_value[T](provider: RankedProviderValue[T]) -> T:
    """Narrow a provider known by the resolver to hold `Found`.

    Returns:
        The provider's coerced value.

    Raises:
        RuntimeError: If an internal accumulating resolver receives a non-found tier.
    """
    result = provider.result
    if isinstance(result, Found):
        return result.value
    msg = f"rank {provider.rank} did not contain a found value"
    raise RuntimeError(msg)


# [해설] 거부 목록 합집합: 약한 계층 항목을 먼저 두고 강한 계층의 새 항목을 뒤에
# [해설]   붙인다(중복 제거). 순서가 보존된다.
def union_lists(lower: list[Any], higher: list[Any]) -> list[Any]:
    """Accumulate two deny-list layers, keeping order and dropping duplicates.

    Shared with the merger so a deny list cannot union in one reader and
    replace in another.

    Returns:
        The lower list followed by the higher entries it does not already hold.
    """
    union = deepcopy(lower)
    for item in higher:
        if item not in union:
            union.append(deepcopy(item))
    return union


# [해설] 거부 목록은 TOML 배열이나 쉼표 문자열 두 형태가 모두 허용되므로 여기서 하나의
# [해설]   리스트로 정규화한다.
def union_entries(value: object) -> list[Any] | None:
    """Normalize one deny-list layer to its entries.

    A deny list may be written as a TOML array or as a comma-separated string
    (`disabled_servers = "a, b"`), and the runtime readers treat the two as
    equivalent — `mcp_disabled._strict_entries` and `model_config._toml_str_list`
    both split on commas. The merge has to accept both spellings too. It did
    not, so a managed string layer was dropped in favor of the user's array and
    the provenance then credited the user's file for a leaf managed policy
    contributes to.

    Returns:
        The trimmed entries, or `None` when the value cannot hold entries.
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return value
    return None


# [해설] 관리형(higher) 표를 사용자(lower) 표 위에 병합하는 공개 함수. 반환값은 병합
# [해설]   결과와 '점으로 이은 leaf -> 출처 이름' 맵이다.
# [해설]   실제 호출자는 service.py의 merge_managed_over_user이며 거기서 union 경로와
# [해설]   leaf 검증기를 채워 넣는다.
def merge_toml_tables(
    lower: Mapping[str, Any],
    higher: Mapping[str, Any],
    *,
    lower_source: str,
    higher_source: str,
    union_paths: frozenset[tuple[str, ...]] = frozenset(),
    higher_leaf_is_valid: Callable[[tuple[str, ...], object], bool] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Deep-merge TOML tables with higher-precedence leaf provenance.

    Args:
        lower: Lower-precedence table.
        higher: Higher-precedence table, whose leaves win.
        lower_source: Source label recorded for surviving `lower` leaves.
        higher_source: Source label recorded for surviving `higher` leaves.
        union_paths: Paths whose lists accumulate instead of being replaced.
            Deny lists must union, because replacing one would be a fail-open.
            Paths match relative to the tables passed here, so a merge of one
            subtree needs them rebased (see `service.union_paths_under`).
        higher_leaf_is_valid: Optional check applied to a `higher` value before
            it displaces a `lower` one. Return `False` to keep the lower value,
            which stops a wrong-typed higher value from discarding a valid
            lower subtree. Receives paths on the same relative basis as
            `union_paths`. Every managed merge passes one, by way of
            `service.merge_managed_over_user`; omitting it leaves the merger
            with no type information, so it displaces only a table that holds
            no nested table.

    Returns:
        Merged table and dotted leaf-to-source mapping.
    """
    merged, provenance = _merge(
        lower,
        higher,
        lower_source=lower_source,
        higher_source=higher_source,
        union_paths=union_paths,
        higher_leaf_is_valid=higher_leaf_is_valid,
    )
    return merged, _dotted(_drop_ancestor_entries(provenance))


def _dotted(provenance: dict[tuple[str, ...], str]) -> dict[str, str]:
    """Join tuple paths for display.

    Provenance is keyed by path tuple everywhere inside this module. TOML allows
    a quoted key that contains dots (`"a.b" = 1` parses to the single key
    `a.b`), so a dotted string is a lossy key: it made `_drop_ancestor_entries`
    delete a live sibling leaf named `a`, and credited the wrong tier for the
    flat key. Joining happens once, here, where the ambiguity is only cosmetic.

    Returns:
        Provenance keyed by dotted path.
    """
    return {".".join(path): source for path, source in provenance.items()}


def _drop_ancestor_entries(
    provenance: dict[tuple[str, ...], str],
) -> dict[tuple[str, ...], str]:
    """Remove entries that are a strict ancestor of another entry.

    A lower empty table that the higher table fills leaves an entry for the
    table itself: it enters the recursion through `lower_provenance`, which
    carries the parent's own path, and the level that fills it never removes it.
    The result claimed a table was a user-controlled leaf alongside the managed
    leaves inside it. A path cannot be both a leaf and a parent, so the ancestor
    is always the stale one.

    Returns:
        Provenance with only leaf entries.
    """
    keys = tuple(provenance)
    return {
        path: source
        for path, source in provenance.items()
        if not any(other[: len(path)] == path and other != path for other in keys)
    }


# [해설][흐름] 각 higher 키마다: 1) 검증기가 거부하면 건너뜀 2) union 경로면 리스트를
# [해설]   합침 3) 양쪽이 표면 재귀
# [해설]   4) 그 외에는 higher 값으로 대체하고 해당 하위 provenance를 higher_source로
# [해설]   갱신한다.
def _merge(
    lower: Mapping[str, Any],
    higher: Mapping[str, Any],
    *,
    lower_source: str,
    higher_source: str,
    union_paths: frozenset[tuple[str, ...]],
    higher_leaf_is_valid: Callable[[tuple[str, ...], object], bool] | None,
    lower_provenance: dict[tuple[str, ...], str] | None = None,
    path_prefix: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[tuple[str, ...], str]]:
    """Recursive half of `merge_toml_tables`.

    Separate so the public signature carries no parameter a caller must not
    pass: `lower_provenance` has to arrive already scoped to `path_prefix`, and
    an unscoped mapping produces wrong provenance with no error.

    Returns:
        Merged table and path-keyed leaf-to-source mapping.
    """
    merged: dict[str, Any] = deepcopy(dict(lower))
    provenance = dict(
        lower_provenance or _leaf_provenance(lower, lower_source, path_prefix)
    )
    for key, value in higher.items():
        path = (*path_prefix, key)
        existing = merged.get(key)
        # [해설][주의] 아래 조건은 '검증기가 없을 때' 스칼라가 표를 덮는 것을 제한한다.
        # [해설]   관리형 병합은 항상 검증기를 넘기므로 실제 정책 경로에서는 검증기 쪽이
        # [해설]   판단한다.
        # A higher scalar must replace a lower table, whatever the table holds.
        # Keeping the table lets a shape collision defeat the higher value.
        # Typed readers then reject the table and use the built-in default.
        #   Example: a user `[threads.relative_time]` table against a managed
        #   `relative_time = false`.
        # Depth is not consulted, so deeper nesting cannot restore the bypass.
        # With `higher_leaf_is_valid`, the check below gates the replacement.
        # That keeps a wrong-typed higher scalar from discarding a valid lower
        # subtree. Without a validator there is no type information here, so
        # only a table that holds no nested table is displaced.
        if (
            isinstance(existing, dict)
            and not isinstance(value, dict)
            and higher_leaf_is_valid is None
            and not _overriding_table_is_scalar_only(existing)
        ):
            continue
        # Validate every managed value at a manifest-backed scalar path,
        # including TOML tables. A table cannot be passed to the validator as
        # a leaf through the recursive branch below, so validating only
        # non-dicts would let `[models.default]` replace a valid string with a
        # dictionary that later runtime readers cannot use.
        if higher_leaf_is_valid is not None and not higher_leaf_is_valid(path, value):
            continue
        if path in union_paths:
            lower_entries = union_entries(existing)
            higher_entries = union_entries(value)
            if lower_entries is not None and higher_entries is None:
                # A higher value that cannot hold names must never replace a
                # deny list: that would drop the lower layer's denials.
                continue
            if lower_entries is not None and higher_entries is not None:
                merged[key] = union_lists(lower_entries, higher_entries)
                provenance[path] = _combined_source(lower_source, higher_source)
                continue
        if isinstance(existing, dict) and isinstance(value, dict):
            nested, nested_provenance = _merge(
                existing,
                value,
                lower_source=lower_source,
                higher_source=higher_source,
                union_paths=union_paths,
                higher_leaf_is_valid=higher_leaf_is_valid,
                lower_provenance={
                    leaf: source
                    for leaf, source in provenance.items()
                    if leaf[: len(path)] == path
                },
                path_prefix=path,
            )
            merged[key] = nested
            # Drop this subtree's old leaves first. A nested merge can delete a
            # leaf (a higher scalar replacing a lower table), and keeping the
            # parent-scope entry would report a path that no longer exists as
            # user-controlled — in the output an administrator reads to audit
            # what policy enforces.
            for leaf in tuple(provenance):
                if leaf[: len(path)] == path:
                    provenance.pop(leaf)
            provenance.update(nested_provenance)
            continue
        merged[key] = deepcopy(value)
        for leaf in tuple(provenance):
            if leaf[: len(path)] == path:
                provenance.pop(leaf)
        provenance.update(_leaf_provenance(value, higher_source, path))
    return merged, provenance


# [해설] 하위에 비어 있지 않은 표가 없는 '스칼라만 든 표'인지 판정한다. 그런 표는 상위
# [해설]   스칼라가 대체해도 잃을 값이 없다.
def _overriding_table_is_scalar_only(table: dict[str, Any]) -> bool:
    """Return `True` when `table` holds no non-empty nested tables at any depth.

    Only direct children need checking: a nested table at any depth makes its
    own parent chain non-empty, so an empty direct child cannot hide one.
    Empty nested tables carry no lower values worth preserving, so they do not
    stop a higher-precedence scalar from replacing the table.
    """
    for child in cast("dict[str, object]", table).values():
        if isinstance(child, dict) and child:
            return False
    return True


def _leaf_provenance(
    value: object, source: str, path: tuple[str, ...]
) -> dict[tuple[str, ...], str]:
    """Return provenance entries for every leaf under `value`."""
    if isinstance(value, dict):
        if not value:
            # An empty table at the root is not a leaf: it would key the whole
            # mapping. Every merge on a machine with no user `config.toml`
            # produced that entry, in the output an administrator reads to audit
            # what policy enforces.
            if not path:
                return {}
            return {path: source}
        result: dict[tuple[str, ...], str] = {}
        for key, child in cast("dict[str, object]", value).items():
            result.update(_leaf_provenance(child, source, (*path, key)))
        return result
    return {path: source}


def _combined_source(lower: str, higher: str) -> str:
    """Combine distinct source labels in precedence order.

    Returns:
        One source or a higher-plus-lower label.
    """
    if lower == higher:
        return higher
    return f"{higher} + {lower}"
